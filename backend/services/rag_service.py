"""
RAG service — wraps the original monolithic RAG logic into clean functions.
All heavy logic (query variants, reranking, confidence, prompts) is preserved
exactly from the original prototype.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from collections import Counter
from functools import lru_cache
from typing import Any, List, Optional, Tuple

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.db_service import get_client, get_collection_name, ensure_collection
from services.ocr_service import perform_ocr_pdf_bytes, perform_ocr_image_bytes

log = logging.getLogger("rag_service")

# ── Config ─────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER            = "./stored_files"
PROCESSED_FILES_REGISTRY = "./processed_files.json"
ENABLE_PDF_OCR_FALLBACK  = True
RETRIEVER_K              = 6
RERANK_TOP_N             = 4
CONFIDENCE_THRESHOLD     = 0.02
SILENCE_THRESHOLD_DB     = -60.0

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Models ──────────────────────────────────────────────────────────────────────
embeddings = OllamaEmbeddings(model="nomic-embed-text")

llm = OllamaLLM(
    model="llama3.2:latest",
    temperature=0.0,
    num_predict=800,
    repeat_penalty=1.15,
    top_k=40,
    top_p=0.90,
)

# ── Vector DB state ─────────────────────────────────────────────────────────────
_vector_db: QdrantVectorStore | None = None
_retriever = None


def _get_vector_db() -> QdrantVectorStore:
    global _vector_db
    if _vector_db is None:
        ensure_collection(embeddings)
        _vector_db = QdrantVectorStore(
            client=get_client(),
            collection_name=get_collection_name(),
            embedding=embeddings,
        )
    return _vector_db


def _refresh_retriever():
    global _retriever, _vector_db
    if _vector_db is None:
        _retriever = None
        return
    _retriever = _vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )


def load_existing_db():
    """Call once at startup to attach to an existing Qdrant collection."""
    global _vector_db
    try:
        client = get_client()
        client.get_collection(get_collection_name())
        _vector_db = QdrantVectorStore(
            client=client,
            collection_name=get_collection_name(),
            embedding=embeddings,
        )
        count = client.get_collection(get_collection_name()).points_count
        log.info(f"Loaded existing DB — {count} vectors")
        _refresh_retriever()
        return True
    except Exception as e:
        log.warning(f"Could not load existing DB: {e}")
        _vector_db = None
        return False


# ── Text utilities (preserved from original) ────────────────────────────────────

AR_STOPWORDS = {
    "ما","ماذا","كيف","هل","في","من","على","الى","إلى","عن","و","او","أو",
    "هو","هي","هم","هذه","هذا","ذلك","تلك","كل","بين","مع","ل","ال","اي","أي",
    "الذي","التي","ثم","بعد","قبل","هناك","هنا","انه","إنه","ان","إن","كان","كانت",
    "دي","ده","دا","يعني","بس","اووي","اوي","كده","كدا",
    "عايز","عاوز","ممكن","لو","لو سمحت","حابب","ابي","ودي","عندي",
    "فيه","فية","فيها","منين","فين","امتى","ليه","ازاي",
}

EN_STOPWORDS = {
    "what","how","is","are","the","a","an","of","to","in","on","for","and",
    "or","between","about","explain","tell","me","does","do","be","types",
    "define","definition","difference","compare","give","show","list","summary",
    "this","that","these","those","with","from","into","by","it","please",
    "can","could","would","should","may","might","will","shall",
    "i","you","we","they","my","your","our","their","its",
}


def _clean(text) -> str:
    return str(text).strip() if text is not None else ""


def detect_language(text: str) -> str:
    text = text or ""

    ar_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    en_chars = len(re.findall(r"[a-zA-Z]", text))

    if ar_chars == 0 and en_chars == 0:
        return "en"

    return "ar" if ar_chars >= en_chars else "en"


def _normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)
    text = re.sub(r"[إأآاٱ]", "ا", text)
    text = re.sub(r"[ىیي]", "ي", text)
    text = re.sub(r"[ؤو]", "و", text)
    text = re.sub(r"[ةه]", "ه", text)
    text = re.sub(r"[ئ]", "ي", text)
    text = re.sub(r"ـ+", "", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize(text: str) -> str:
    if not text:
        return ""
    if detect_language(text) == "ar":
        return _normalize_arabic(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _keywords(text: str, lang: str) -> List[str]:
    words = _normalize(text).split()
    stops = AR_STOPWORDS if lang == "ar" else EN_STOPWORDS
    return [w for w in words if w not in stops and len(w) > 2]


def _ngrams(text: str, n: int = 2) -> List[str]:
    words = _normalize(text).split()
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)] if len(words) >= n else []


def _is_meaningful(text: str) -> bool:
    c = _clean(text)
    return len(c) >= 15 and bool(re.search(r"[\u0600-\u06FFa-zA-Z]", c))


def _deduplicate(docs: List[Document]) -> List[Document]:
    seen, out = set(), []
    for d in docs:
        key = _clean(d.page_content)[:1000]
        if key and key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _deduplicate_retrieved(docs: List[Document]) -> List[Document]:
    seen, out = set(), []
    for d in docs:
        key = (_clean(d.page_content)[:700], d.metadata.get("source",""), d.metadata.get("page",-1))
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _enrich(text: str) -> str:
    text = _clean(text)
    if not text:
        return ""
    lang   = detect_language(text)
    blocks = [text]
    if lang == "ar":
        norm = _normalize_arabic(text)
        if norm and norm != text:
            blocks += ["[normalized_arabic]", norm]
    else:
        lower = text.lower()
        if lower != text:
            blocks += ["[lowercase]", lower]
    return "\n\n".join(b for b in blocks if _clean(b))


# ── Translation & query variants ───────────────────────────────────────────────

@lru_cache(maxsize=512)
def _translate(text: str, target_lang: str) -> str:
    text = _clean(text)
    if not text:
        return ""
    prompt = (
        f"Translate this Arabic text to English. Return ONLY the translation:\n{text}"
        if target_lang == "en"
        else f"Translate this English text to Arabic. Return ONLY the translation:\n{text}"
    )
    try:
        out = str(llm.invoke(prompt)).strip()
        return _clean(re.split(r"\n\n", out)[0])
    except Exception as e:
        log.warning(f"Translation failed: {e}")
        return ""


@lru_cache(maxsize=256)
def _rephrase(query: str, lang: str) -> str:
    if not query:
        return ""
    prompt = (
        f"Rephrase this question differently using synonyms (one sentence only):\n{query}"
        if lang == "en"
        else f"أعِد صياغة هذا السؤال بأسلوب مختلف (جملة واحدة فقط):\n{query}"
    )
    try:
        return _clean(str(llm.invoke(prompt)).strip().split("\n")[0])
    except Exception:
        return ""

def _is_mixed_language(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or "")) and bool(re.search(r"[a-zA-Z]", text or ""))

def _query_variants(question: str, lang: str) -> List[str]:
    q = _clean(question)

    if not q:
        return []

    variants = []

    def add(x: str):
        x = _clean(x)
        if x and x not in variants:
            variants.append(x)

    add(q)

    if _is_mixed_language(q):
        add(_normalize_arabic(q))
        add(q.lower())

        tr_en = _translate(q, "en")
        tr_ar = _translate(q, "ar")

        add(tr_en)
        add(tr_en.lower())
        add(tr_ar)
        add(_normalize_arabic(tr_ar))

    elif detect_language(q) == "ar":
        add(_normalize_arabic(q))

        tr_en = _translate(q, "en")
        add(tr_en)
        add(tr_en.lower())

        rp_en = _rephrase(tr_en, "en") if tr_en else ""
        add(rp_en)

    else:
        add(q.lower())

        tr_ar = _translate(q, "ar")
        add(tr_ar)
        add(_normalize_arabic(tr_ar))

        rp_ar = _rephrase(tr_ar, "ar") if tr_ar else ""
        add(rp_ar)
        add(_normalize_arabic(rp_ar))

    return variants[:12]


# ── Lexical scoring ────────────────────────────────────────────────────────────

def _lex_score(query: str, doc_text: str, lang: str) -> float:
    kws = _keywords(query, lang)
    if not kws:
        return 0.0
    doc_norm     = _normalize(doc_text)
    uni_score    = sum(1 for kw in kws if kw in doc_norm) / max(len(kws), 1)
    bigs         = _ngrams(query, 2)
    bigram_score = (sum(1 for bg in bigs if bg in doc_norm) / max(len(bigs), 1)) * 1.5 if bigs else 0.0
    return min(uni_score + bigram_score, 1.0)


def _confidence(question: str, docs: List[Document], lang: str) -> float:
    if not docs:
        return 0.0

    variants = _query_variants(question, lang)
    scores = []

    for qv in variants:
        q_lang = detect_language(qv)
        kws = _keywords(qv, q_lang)

        if not kws:
            continue

        for doc in docs[:RERANK_TOP_N]:
            content = _normalize(_clean(doc.page_content))
            score = sum(1 for kw in kws if kw in content) / len(kws)
            scores.append(score)

    if not scores:
        return 0.05

    best = max(scores)
    bonus = 0.03 if best > 0 else 0.0

    return min(best + bonus, 1.0)


# ── Reranking ──────────────────────────────────────────────────────────────────

def _rerank(variants: List[str], docs: List[Document]) -> Tuple[List[Document], List[dict]]:
    scored = []
    for idx, d in enumerate(docs):
        content = _clean(d.page_content)
        if not _is_meaningful(content):
            continue
        max_lex   = 0.0
        best_var  = ""
        for qv in variants:
            s = _lex_score(qv, content, detect_language(qv))
            if s > max_lex:
                max_lex, best_var = s, qv

        head        = _normalize(content[:300])
        bonus_head  = sum(0.05 for qv in variants[:3] for kw in _keywords(qv, detect_language(qv)) if kw in head)
        bonus_bi    = sum(0.08 for qv in variants[:2] for bg in _ngrams(qv, 2) if bg in _normalize(content))
        final_score = max_lex + bonus_head + bonus_bi - idx * 0.002

        scored.append((final_score, d, {
            "source":  d.metadata.get("source", "?"),
            "page":    d.metadata.get("page", 0),
            "score":   round(final_score, 4),
            "preview": content[:120].replace("\n", " "),
        }))

    scored.sort(key=lambda x: x[0], reverse=True)
    ranked = [d for s, d, _ in scored if s > 0]
    debugs = [dbg for s, _, dbg in scored if s > 0]

    if not ranked:
        ranked = [d for _, d, _ in scored[:RERANK_TOP_N]]
        debugs = [dbg for _, _, dbg in scored[:RERANK_TOP_N]]

    return ranked[:RERANK_TOP_N], debugs[:RERANK_TOP_N]


def _retrieve(question: str, lang: str) -> Tuple[List[Document], str]:
    if _retriever is None:
        return [], "retriever is None"

    variants = _query_variants(question, lang)

    all_docs: List[Document] = []

    for q in variants:
        try:
            docs = _retriever.invoke(q)
            all_docs.extend(docs)
        except Exception as e:
            log.error(f"Retrieval error for '{q}': {e}")

    all_docs = _deduplicate_retrieved(all_docs)

    ranked, debugs = _rerank(variants, all_docs)

    # fallback: لو rerank رجع فاضي، استخدم أول docs راجعة من vector search
    if not ranked and all_docs:
        ranked = all_docs[:RERANK_TOP_N]
        debugs = [{
            "source": d.metadata.get("source", "?"),
            "page": d.metadata.get("page", 0),
            "score": 0,
            "preview": _clean(d.page_content)[:120].replace("\n", " "),
        } for d in ranked]

    debug_str = "\n".join(
        f"Rank {i+1}: {d['source']} p{d['page']} score={d['score']} | {d['preview'][:60]}"
        for i, d in enumerate(debugs)
    )

    return ranked[:RERANK_TOP_N], debug_str


# ── Prompt & cleanup ───────────────────────────────────────────────────────────

def build_prompt(context: str, question: str, lang: str) -> str:
    if lang == "ar":
        return f"""أنت نظام استخراج معلومات متقدم. مهمتك: استخرج إجابة كاملة ومفصّلة من السياق المقدم.

**قواعد الإجابة — يجب اتباعها بدقة:**

1. اقرأ السياق كاملاً قبل الكتابة.
2. إذا وُجدت المعلومة كاملةً في السياق:
   - قدّم إجابة واضحة ومفصّلة (3 جمل على الأقل إذا أمكن).
   - استخدم فقرات منظّمة أو نقاط (•) عند الحديث عن قوائم أو خطوات.
   - اذكر الأرقام والتواريخ والأسماء كما وردت حرفياً في السياق.
3. إذا وُجدت المعلومة جزئياً في السياق:
   - أجب بما هو موجود فعلاً، ثم وضّح باختصار ما لم يُذكر في الملفات.
4. إذا لم تُوجد المعلومة إطلاقاً:
   - قل فقط: "المعلومة غير موجودة في الملفات المرفوعة."
5. لا تبدأ بعبارات مثل "بناءً على السياق..." أو "وفقاً للمعلومات...".
6. لا تكرر السؤال في الإجابة.
7. لا تستخدم أي معرفة خارجية.
8. الإجابة بالعربية الفصحى الواضحة.
9. بعد الإجابة، قدّم مثال بسيط يوضح الفكرة.
10. استخدم تنسيق:
    - الشرح:
    - المثال:
11. قد يكون السؤال بالعربية والسياق بالإنجليزية أو العكس. افهم المعنى بين اللغتين وأجب بلغة السؤال الأساسية.
12. إذا كان السؤال يحتوي على عربي وإنجليزي معًا، اعتبر المصطلحات الإنجليزية جزءًا من السؤال ولا تترجمها ترجمة خاطئة.

**السياق:**
{context}

**السؤال:**
{question}

**الإجابة الكاملة:**"""

    return f"""You are an advanced information extraction system. Your task: extract a complete, well-structured answer from the provided context.

**Answering rules — follow precisely:**

1. Read the entire context before writing.
2. If the information is fully present in the context:
   - Provide a clear, detailed answer (at least 3 sentences when possible).
   - Use organized paragraphs or bullet points (•) for lists or step-by-step content.
   - State numbers, dates, and names exactly as they appear in the context.
3. If the information is only partially present:
   - Answer with what IS in the context, then briefly note what is missing from the documents.
4. If the information is entirely absent:
   - Say only: "The information is not available in the uploaded files."
5. Do NOT open with "Based on the context..." or "According to the information...".
6. Do NOT repeat the question in the answer.
7. Do NOT use any knowledge from outside the context below.
8. Answer in clear, professional English.
9. After the answer, provide a simple example.
10. Use format:
   - Explanation:
   - Example:
11. The question and context may be in different languages. Understand the meaning across Arabic and English.
12. If the question mixes Arabic and English, preserve technical English terms and answer in the dominant language of the question.

**Context:**
{context}

**Question:**
{question}

**Complete Answer:**"""


def _clean_answer(text: str, lang: str) -> str:
    text = _clean(text)
    banned = [
        "Sources:","Source:","المصادر:","المصدر:",
        "Based on the context,","According to the context,",
        "بناءً على السياق،","وفقاً للمعلومات المتاحة،",
    ]
    lines, seen = [], []
    for line in text.splitlines():
        for b in banned:
            if line.strip().startswith(b):
                line = line.replace(b, "", 1).strip()
                break
        key = _normalize(line)
        if key and key in seen:
            continue
        lines.append(line)
        if key:
            seen.append(key)
            if len(seen) > 10:
                seen.pop(0)
    result = "\n".join(lines).strip()
    if not result:
        return ("المعلومة غير موجودة في الملفات المرفوعة." if lang == "ar"
                else "The information is not available in the uploaded files.")
    return result


def _build_sources(docs: List[Document], lang: str) -> str:
    if not docs:
        return ""
    counts: Counter = Counter()
    pages:  dict    = {}
    for d in docs:
        src = os.path.basename(d.metadata.get("source", "?"))
        pg  = d.metadata.get("page", 0)
        counts[src] += 1
        pages.setdefault(src, set()).add(pg + 1 if isinstance(pg, int) else pg)
    parts = []
    for src, _ in counts.most_common(3):
        pg_str = ", ".join(map(str, sorted(list(pages.get(src, [])))[:3]))
        parts.append(f"{src} (p. {pg_str})" if pg_str else src)
    prefix = "المصادر: " if lang == "ar" else "Sources: "
    return prefix + " | ".join(parts)


# ── File registry ──────────────────────────────────────────────────────────────

def _load_registry() -> dict:
    if os.path.isfile(PROCESSED_FILES_REGISTRY):
        try:
            with open(PROCESSED_FILES_REGISTRY, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_registry(reg: dict):
    try:
        with open(PROCESSED_FILES_REGISTRY, "w", encoding="utf-8") as f:
            json.dump(reg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"Registry save error: {e}")


def _file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _safe_filename(filename: str) -> str:
    name = os.path.basename(filename or "unknown")
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name.strip() or "unknown"


def _save_uploaded_file(filename: str, data: bytes, fhash: str) -> str:
    safe_name = _safe_filename(filename)
    stem, ext = os.path.splitext(safe_name)

    stored_name = f"{stem}_{fhash[:10]}{ext}"
    stored_path = os.path.join(UPLOAD_FOLDER, stored_name)

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    with open(stored_path, "wb") as f:
        f.write(data)

    return stored_path


def list_stored_files() -> list[dict]:
    registry = _load_registry()
    files = []

    for _, info in registry.items():
        files.append({
            "filename": info.get("filename"),
            "stored_path": info.get("stored_path"),
            "file_type": info.get("file_type"),
            "chunks": info.get("chunks", 0),
            "processed_at": info.get("processed_at"),
        })

    files.sort(key=lambda x: x.get("processed_at") or "", reverse=True)
    return files

def _get_file_type(filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    return {"pdf":"pdf","docx":"docx","doc":"doc","txt":"txt",
            "md":"markdown","json":"json","jpg":"image",
            "jpeg":"image","png":"image"}.get(ext, "unknown")


# ── Document loading from bytes ────────────────────────────────────────────────

def _load_document_from_bytes(filename: str, data: bytes) -> List[Document]:
    """Load, OCR if needed, and enrich a document from its raw bytes."""
    ext      = filename.lower().rsplit(".", 1)[-1]
    filetype = _get_file_type(filename)
    ts       = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta_base = {"source": filename, "file_type": filetype, "page": 0, "timestamp": ts}

    raw_docs: List[Document] = []

    try:
        if ext == "pdf":
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                tf.write(data)
                tmp_path = tf.name
            try:
                loader    = PyPDFLoader(tmp_path)
                raw_docs  = loader.load()
                text_body = "".join(d.page_content for d in raw_docs).strip()
                if ENABLE_PDF_OCR_FALLBACK and len(text_body) < 20:
                    ocr_text = perform_ocr_pdf_bytes(data)
                    if _clean(ocr_text):
                        raw_docs = [Document(page_content=ocr_text, metadata={**meta_base, "ocr_fallback": True})]
                    else:
                        raw_docs = []
                else:
                    for i, d in enumerate(raw_docs):
                        d.metadata = {**meta_base, "page": d.metadata.get("page", i)}
            finally:
                os.unlink(tmp_path)

        elif ext in {"docx", "doc"}:
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tf:
                tf.write(data)
                tmp_path = tf.name
            try:
                raw_docs = Docx2txtLoader(tmp_path).load()
                for d in raw_docs:
                    d.metadata = {**meta_base}
            finally:
                os.unlink(tmp_path)

        elif ext in {"txt", "md"}:
            text = data.decode("utf-8", errors="replace").strip()
            raw_docs = [Document(page_content=text, metadata={**meta_base})]

        elif ext == "json":
            try:
                obj  = json.loads(data.decode("utf-8", errors="replace"))
                text = json.dumps(obj, ensure_ascii=False, indent=2)
            except Exception:
                text = data.decode("utf-8", errors="replace")
            raw_docs = [Document(page_content=text, metadata={**meta_base})]

        elif ext in {"jpg", "jpeg", "png", "tiff", "bmp", "webp"}:
            text = perform_ocr_image_bytes(data)
            if _clean(text):
                raw_docs = [Document(page_content=text, metadata={**meta_base, "ocr": True})]

    except Exception as e:
        log.error(f"load_document error ({filename}): {e}")

    enriched = []
    for d in raw_docs:
        content = _enrich(d.page_content)
        if content:
            d.page_content = content
            enriched.append(d)

    log.info(f"Loaded '{filename}' → {len(enriched)} docs")
    return enriched


# ── Public API ─────────────────────────────────────────────────────────────────

def update_db_files(files: List[dict[str, Any]]) -> int:
    """
    Ingest a list of {'filename': str, 'data': bytes} dicts into Qdrant.
    Also saves uploaded files physically inside ./stored_files.
    Returns total number of chunks added.
    """
    global _vector_db

    registry = _load_registry()
    new_files = []
    skipped = []

    for f in files:
        filename = f.get("filename") or "unknown"
        data = f.get("data") or b""

        if not data:
            log.warning(f"Skip empty file: {filename}")
            continue

        h = _file_hash(data)

        if h in registry:
            skipped.append(filename)
            log.info(f"Skip already processed: {filename}")
        else:
            new_files.append((filename, data, h))

    if not new_files:
        log.info(f"No new files — skipped: {skipped}")
        return 0

    all_docs: List[Document] = []
    per_file_info: dict = {}

    for filename, data, fhash in new_files:
        stored_path = _save_uploaded_file(filename, data, fhash)

        docs = _load_document_from_bytes(filename, data)

        for d in docs:
            d.metadata["stored_path"] = stored_path

        all_docs.extend(docs)

        per_file_info[filename] = {
            "hash": fhash,
            "stored_path": stored_path,
            "docs_count": len(docs),
        }

    all_docs = _deduplicate(all_docs)

    if not all_docs:
        log.warning("No valid documents extracted from uploaded files.")
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        separators=["\n\n\n", "\n\n", "\n", ".", " ", ""],
    )

    chunks = [
        c for c in splitter.split_documents(all_docs)
        if _is_meaningful(c.page_content)
    ]

    if not chunks:
        log.warning("No meaningful chunks generated.")
        return 0

    cps: Counter = Counter(c.metadata.get("source", "?") for c in chunks)

    idx_map: Counter = Counter()

    for chunk in chunks:
        src = chunk.metadata.get("source", "?")
        chunk.metadata["chunk_index"] = idx_map[src]
        chunk.metadata["total_chunks"] = cps[src]
        idx_map[src] += 1

    ensure_collection(embeddings)

    vdb = _get_vector_db()
    vdb.add_documents(chunks)

    _refresh_retriever()

    for filename, info in per_file_info.items():
        fhash = info["hash"]

        registry[fhash] = {
            "filename": filename,
            "stored_path": info["stored_path"],
            "file_type": _get_file_type(filename),
            "chunks": cps.get(filename, 0),
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    _save_registry(registry)

    log.info(
        f"Added {len(chunks)} chunks from {len(new_files)} new file(s). "
        f"Skipped: {skipped}"
    )

    return len(chunks)


def ask_question(query: str, lang: str = "auto") -> dict:
    """
    Answer a question using RAG.
    Returns {"answer": str, "sources": str}
    """
    if _retriever is None:
        load_existing_db()
    if _retriever is None:
        return {
            "answer": "⚠️ Database is empty. Please upload files first.",
            "sources": "",
        }

    detected_lang = detect_language(query) if lang == "auto" else lang
    docs, debug   = _retrieve(query, detected_lang)
    log.debug(f"Retrieval:\n{debug}")

    if not docs:
        no_info = ("المعلومة غير موجودة في الملفات المرفوعة."
                   if detected_lang == "ar"
                   else "The information is not available in the uploaded files.")
        return {"answer": no_info, "sources": ""}

    context_parts = [
        f"[Chunk {i+1} | {d.metadata.get('source','?')} | page {d.metadata.get('page',0)}]\n{d.page_content}"
        for i, d in enumerate(docs)
    ]
    context = "\n\n---\n\n".join(context_parts)

    #if _confidence(query, docs, detected_lang) < CONFIDENCE_THRESHOLD:
     #   no_info = ("المعلومة غير موجودة في الملفات المرفوعة."
      #             if detected_lang == "ar"
       #            else "The information is not available in the uploaded files.")
        #return {"answer": no_info, "sources": _build_sources(docs, detected_lang)}

    prompt = build_prompt(context, query, detected_lang)
    try:
        t0     = time.time()
        answer = str(llm.invoke(prompt))
        log.info(f"LLM answered in {time.time()-t0:.2f}s")
        answer = _clean_answer(answer, detected_lang)
    except Exception as e:
        log.error(f"LLM error: {e}")
        answer = f"Error generating answer: {e}"

    return {"answer": answer, "sources": _build_sources(docs, detected_lang)}
