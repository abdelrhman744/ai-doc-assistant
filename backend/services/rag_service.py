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


def _loose_arabic(text: str) -> str:
    """
    Extra-tolerant Arabic normalization for typo-heavy user questions.
    Examples:
    - كرة / كره
    - أركان / اركان
    - الإسلام / الاسلام
    """
    text = _normalize_arabic(text)
    text = re.sub(r"\bال", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _loose_english(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=256)
def _fix_query_spelling(query: str, lang: str) -> str:
    """
    Lightweight LLM spelling correction.
    Used only to create extra retrieval variants; it never replaces the user's original question.
    """
    query = _clean(query)
    if not query:
        return ""

    if lang == "ar":
        prompt = f"""صحح الأخطاء الإملائية في السؤال التالي بدون تغيير المعنى.
أعد السؤال فقط بدون شرح أو مقدمات:

{query}"""
    else:
        prompt = f"""Correct spelling mistakes in this question without changing its meaning.
Return only the corrected question, no explanation:

{query}"""

    try:
        out = str(llm.invoke(prompt)).strip()
        return _clean(out.splitlines()[0])
    except Exception as e:
        log.warning(f"Spelling correction failed: {e}")
        return ""


def _query_variants(question: str, lang: str) -> List[str]:
    """
    Build robust retrieval variants:
    - original query
    - normalized Arabic / lowercase English
    - typo-corrected query
    - Arabic ↔ English translations
    - rephrases
    """
    q = _clean(question)
    if not q:
        return []

    variants: List[str] = []

    def add(x: str):
        x = _clean(x)
        if x and x not in variants:
            variants.append(x)

    detected = detect_language(q)

    add(q)
    add(_normalize(q))

    if detected == "ar" or _is_mixed_language(q):
        add(_normalize_arabic(q))
        add(_loose_arabic(q))
    else:
        add(q.lower())
        add(_loose_english(q))

    fixed = _fix_query_spelling(q, detected)
    if fixed:
        add(fixed)
        add(_normalize(fixed))
        if detect_language(fixed) == "ar":
            add(_normalize_arabic(fixed))
            add(_loose_arabic(fixed))
        else:
            add(fixed.lower())
            add(_loose_english(fixed))

    # Always try Arabic -> English
    tr_en = _translate(q, "en")
    if tr_en:
        add(tr_en)
        add(tr_en.lower())
        add(_loose_english(tr_en))

        fixed_en = _fix_query_spelling(tr_en, "en")
        if fixed_en:
            add(fixed_en)
            add(fixed_en.lower())
            add(_loose_english(fixed_en))

        rp_en = _rephrase(tr_en, "en")
        if rp_en:
            add(rp_en)
            add(rp_en.lower())
            add(_loose_english(rp_en))

    # Always try English -> Arabic
    tr_ar = _translate(q, "ar")
    if tr_ar:
        add(tr_ar)
        add(_normalize_arabic(tr_ar))
        add(_loose_arabic(tr_ar))

        fixed_ar = _fix_query_spelling(tr_ar, "ar")
        if fixed_ar:
            add(fixed_ar)
            add(_normalize_arabic(fixed_ar))
            add(_loose_arabic(fixed_ar))

        rp_ar = _rephrase(tr_ar, "ar")
        if rp_ar:
            add(rp_ar)
            add(_normalize_arabic(rp_ar))
            add(_loose_arabic(rp_ar))

    return variants[:18]


# ── Lexical scoring# ── Lexical scoring ────────────────────────────────────────────────────────────

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
    """
    Rerank without throwing away vector-search results.
    This is important for Arabic ↔ English cross-language questions and typo-heavy queries.
    """
    scored = []

    for idx, d in enumerate(docs):
        content = _clean(d.page_content)
        if not _is_meaningful(content):
            continue

        max_lex = 0.0

        for qv in variants:
            q_lang = detect_language(qv)
            s = _lex_score(qv, content, q_lang)

            if q_lang == "ar":
                loose_q = _loose_arabic(qv)
                loose_doc = _loose_arabic(content)
                kws = [w for w in loose_q.split() if len(w) > 2]
                if kws:
                    loose_score = sum(1 for kw in kws if kw in loose_doc) / len(kws)
                    s = max(s, loose_score)
            else:
                loose_q = _loose_english(qv)
                loose_doc = _loose_english(content)
                kws = [w for w in loose_q.split() if len(w) > 2]
                if kws:
                    loose_score = sum(1 for kw in kws if kw in loose_doc) / len(kws)
                    s = max(s, loose_score)

            max_lex = max(max_lex, s)

        # Keep vector-search ordering as fallback even if lexical score is low.
        final_score = max_lex - idx * 0.001

        scored.append((final_score, d, {
            "source": d.metadata.get("source", "?"),
            "page": d.metadata.get("page", 0),
            "score": round(final_score, 4),
            "preview": content[:120].replace("\n", " "),
        }))

    scored.sort(key=lambda x: x[0], reverse=True)

    ranked = [d for _, d, _ in scored]
    debugs = [dbg for _, _, dbg in scored]

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

    # Fallback for very typo-heavy or short queries
    if not all_docs:
        simplified = " ".join(
            re.findall(r"[\u0600-\u06FFa-zA-Z]{2,}", _normalize(question))[:8]
        )
        if simplified:
            try:
                all_docs.extend(_retriever.invoke(simplified))
            except Exception as e:
                log.error(f"Fallback retrieval error: {e}")

    all_docs = _deduplicate_retrieved(all_docs)

    if not all_docs:
        return [], "no docs retrieved"

    ranked, debugs = _rerank(variants, all_docs)

    debug_str = "\n".join(
        f"Rank {i+1}: {d['source']} p{d['page']} score={d['score']} | {d['preview'][:80]}"
        for i, d in enumerate(debugs)
    )

    return ranked[:RERANK_TOP_N], debug_str


# ── Prompt & cleanup# ── Prompt & cleanup ───────────────────────────────────────────────────────────

def build_prompt(context: str, question: str, lang: str) -> str:
    if lang == "ar":
        return f"""أنت نظام استخراج معلومات متقدم. مهمتك: استخرج إجابة كاملة ومفصّلة من السياق المقدم فقط.

**قواعد الإجابة — يجب اتباعها بدقة:**

1. اقرأ السياق كاملاً قبل الكتابة.
2. السؤال قد يكون بالعربية والسياق بالإنجليزية أو العكس؛ افهم المعنى بين اللغتين.
3. السؤال قد يحتوي على أخطاء إملائية أو حروف ناقصة أو كلمات عامية؛ حاول فهم المقصود من السياق.
4. إذا وجدت معلومة مرتبطة أو قريبة جدًا من السؤال في السياق، أجب بها ولا تقل إن المعلومة غير موجودة.
5. لا تقل "المعلومة غير موجودة في الملفات المرفوعة" إلا إذا كان السياق لا يحتوي على أي معلومة مرتبطة بالسؤال نهائيًا.
6. لا تستخدم أي معرفة خارجية خارج السياق.
7. لا تبدأ بعبارات مثل "بناءً على السياق" أو "وفقاً للمعلومات".
8. لا تكرر السؤال في الإجابة.
9. اذكر الأرقام والتواريخ والأسماء كما وردت في السياق.
10. الإجابة بالعربية الواضحة، مع الحفاظ على المصطلحات الإنجليزية المهمة كما هي.
11. بعد الإجابة، قدم مثالًا بسيطًا إذا كان مناسبًا.
12. استخدم تنسيق:
    - الشرح:
    - المثال:

**السياق:**
{context}

**السؤال:**
{question}

**الإجابة الكاملة:**"""

    return f"""You are an advanced information extraction system. Your task is to extract a complete, well-structured answer from the provided context only.

**Answering rules — follow precisely:**

1. Read the entire context before writing.
2. The question and context may be in different languages. Understand the meaning across Arabic and English.
3. The question may contain spelling mistakes, missing letters, dialect words, or mixed Arabic/English terms. Infer the intended meaning from the context.
4. If the context contains related or very close information, answer using it. Do not say it is unavailable.
5. Only say "The information is not available in the uploaded files." if the context has no related information at all.
6. Do not use external knowledge outside the context.
7. Do not open with "Based on the context" or "According to the information".
8. Do not repeat the question.
9. State numbers, dates, and names exactly as they appear in the context.
10. Answer in clear English, preserving important Arabic or English technical terms when needed.
11. After the answer, provide a simple example if useful.
12. Use format:
   - Explanation:
   - Example:

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


def generate_quiz(
    topic: str = "",
    num_questions: int = 5,
    question_type: str = "mixed",
    lang: str = "auto",
) -> dict:
    """
    Generate a quiz from uploaded documents using existing Qdrant retriever.
    Works with Arabic, English, mixed questions, and minor spelling mistakes.
    """

    if _retriever is None:
        load_existing_db()

    if _retriever is None:
        return {
            "quiz": "⚠️ Database is empty. Please upload files first.",
            "sources": "",
        }

    topic = _clean(topic)
    num_questions = max(1, min(int(num_questions), 20))

    detected_lang = detect_language(topic) if lang == "auto" and topic else lang
    if detected_lang == "auto":
        detected_lang = "en"

    search_query = topic if topic else "summary key concepts main ideas important information"

    docs, debug = _retrieve(search_query, detected_lang)
    log.debug(f"Quiz retrieval:\n{debug}")

    if not docs:
        msg = (
            "المعلومة غير موجودة في الملفات المرفوعة."
            if detected_lang == "ar"
            else "The information is not available in the uploaded files."
        )
        return {"quiz": msg, "sources": ""}

    context_parts = [
        f"[Chunk {i+1} | {d.metadata.get('source','?')} | page {d.metadata.get('page',0)}]\n{d.page_content}"
        for i, d in enumerate(docs)
    ]

    context = "\n\n---\n\n".join(context_parts)

    if detected_lang == "ar":
        prompt = f"""
أنت منشئ اختبارات تعليمي محترف.

مهمتك: أنشئ اختبارًا من الملفات المرفوعة فقط.

القواعد:
1. استخدم المعلومات الموجودة في السياق فقط.
2. لا تستخدم معرفة خارجية.
3. الموضوع قد يكون مكتوبًا بالعربية والسياق بالإنجليزية أو العكس؛ افهم المعنى بين اللغتين.
4. الموضوع قد يحتوي على أخطاء إملائية بسيطة؛ استنتج المقصود من السياق.
5. إذا وجدت معلومات قريبة من الموضوع في السياق، أنشئ الاختبار منها ولا تقل إن المعلومة غير موجودة.
6. عدد الأسئلة المطلوب: {num_questions}.
7. نوع الأسئلة: {question_type}.
8. إذا كان النوع mcq، اجعل كل الأسئلة اختيارًا من متعدد.
9. إذا كان النوع true_false، اجعل كل الأسئلة صح أو خطأ.
10. إذا كان النوع short_answer، اجعل كل الأسئلة إجابة قصيرة.
11. إذا كان النوع mixed، اخلط بين اختيار من متعدد وصح/خطأ وإجابة قصيرة.
12. في أسئلة الاختيار من متعدد، اكتب 4 اختيارات A, B, C, D.
13. بعد كل سؤال، اكتب Answer ثم Explanation.
14. اكتب بالعربية الواضحة، واترك المصطلحات الإنجليزية المهمة كما هي.

السياق:
{context}

الموضوع المطلوب:
{topic or "اختبار عام من الملفات"}

اكتب الاختبار بهذا الشكل:

# Quiz

## Question 1
...

Answer:
...

Explanation:
...
"""
    else:
        prompt = f"""
You are a professional educational quiz generator.

Your task: create a quiz from the uploaded documents only.

Rules:
1. Use only the provided context.
2. Do not use external knowledge.
3. The topic may be Arabic while the context is English, or the opposite. Understand meaning across languages.
4. The topic may contain minor spelling mistakes. Infer the intended meaning from the context.
5. If the context contains related information, create the quiz from it. Do not say it is unavailable.
6. Number of questions: {num_questions}.
7. Question type: {question_type}.
8. If the type is mcq, make all questions multiple choice.
9. If the type is true_false, make all questions True / False.
10. If the type is short_answer, make all questions short answer.
11. If the type is mixed, include multiple choice, True / False, and short answer.
12. For MCQs, provide 4 options: A, B, C, D.
13. After each question, provide Answer and Explanation.
14. Answer in clear English, preserving important Arabic or English technical terms when needed.

Context:
{context}

Requested topic:
{topic or "General quiz from uploaded files"}

Format:

# Quiz

## Question 1
...

Answer:
...

Explanation:
...
"""

    try:
        answer = str(llm.invoke(prompt)).strip()
        answer = _clean_answer(answer, detected_lang)
    except Exception as e:
        log.error(f"Quiz generation error: {e}")
        answer = f"Error generating quiz: {e}"

    return {
        "quiz": answer,
        "sources": _build_sources(docs, detected_lang),
    }
