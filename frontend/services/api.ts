const BASE = "/api";

export interface ChatResponse {
  answer:   string;
  sources:  string;
  stt_text: string;
}

export interface UploadResponse {
  message:      string;
  chunks_added: number;
}

export async function askQuestion(
  query:    string,
  language: "auto" | "ar" | "en" = "auto"
): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/chat`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ query, language }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Chat request failed");
  }
  return res.json();
}

export async function askVoice(
  audioBlob: Blob,
  language:  string = "auto"
): Promise<ChatResponse> {
  const form = new FormData();
  form.append("audio", audioBlob, "recording.webm");
  form.append("language", language);

  const res = await fetch(`${BASE}/chat/voice`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Voice request failed");
  }
  return res.json();
}

export async function uploadFiles(files: File[]): Promise<UploadResponse> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));

  const res = await fetch(`${BASE}/upload`, { method: "POST", body: form });
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
