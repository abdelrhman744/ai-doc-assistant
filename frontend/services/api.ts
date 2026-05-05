const BASE = "/api";

export type Language = "auto" | "ar" | "en";

export interface ChatResponse {
  answer: string;
  sources: string;
  stt_text: string;
}

export interface UploadResponse {
  message: string;
  chunks_added: number;
  stored_files?: any[];
}

export async function askQuestion(
  query: string,
  language: Language = "auto"
): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, language }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Chat request failed");
  }

  return res.json();
}

export async function askVoice(
  audioBlob: Blob,
  language: Language = "auto"
): Promise<ChatResponse> {
  const form = new FormData();
  form.append("audio", audioBlob, "recording.webm");
  form.append("language", language);

  const res = await fetch(`${BASE}/chat/voice`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Voice request failed");
  }

  return res.json();
}

export async function uploadFiles(files: File[]): Promise<UploadResponse> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));

  const res = await fetch(`${BASE}/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Upload failed");
  }

  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

export interface QuizRequest {
  topic: string;
  num_questions: number;
  question_type: "mcq" | "true_false" | "short_answer" | "mixed";
  language: Language;
}

export interface QuizResponse {
  quiz: string;
  sources: string;
}

export async function generateQuiz(
  payload: QuizRequest
): Promise<QuizResponse> {
  const res = await fetch(`${BASE}/quiz/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Quiz generation failed");
  }

  return res.json();
}