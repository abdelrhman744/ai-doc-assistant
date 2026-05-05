"use client";

import { useEffect, useState } from "react";
import { FileQuestion, MessageSquareText } from "lucide-react";
import ChatBox from "@/components/ChatBox";
import QuizGenerator from "@/components/QuizGenerator";
import UploadBox from "@/components/UploadBox";
import { checkHealth } from "@/services/api";

type Section = "chat" | "quiz";

export default function Home() {
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [uploadNote, setUploadNote] = useState<string>("");
  const [section, setSection] = useState<Section>("chat");

  useEffect(() => {
    checkHealth().then(setHealthy);
  }, []);

  return (
    <div className="flex h-screen overflow-hidden bg-paper" dir="ltr">
      <aside
        className={`
          flex flex-col border-r border-ash bg-white
          transition-all duration-300 ease-in-out flex-shrink-0
          ${sidebarOpen ? "w-72" : "w-0 overflow-hidden"}
        `}
      >
        <div className="flex flex-col h-full p-5 gap-6 min-w-[18rem]">
          <div className="flex items-center gap-2.5 pt-1">
            <div className="w-8 h-8 rounded-lg bg-ink flex items-center justify-center">
              <MessageSquareText className="w-4 h-4 text-paper" />
            </div>
            <span className="font-semibold text-ink">DocAssist AI</span>
          </div>

          <div className="flex items-center gap-2 text-xs">
            <span
              className={`w-2 h-2 rounded-full ${
                healthy === null
                  ? "bg-muted animate-pulse"
                  : healthy
                  ? "bg-teal"
                  : "bg-accent"
              }`}
            />
            <span className="text-muted">
              {healthy === null
                ? "Connecting…"
                : healthy
                ? "Backend online"
                : "Backend offline"}
            </span>
          </div>

          <div>
            <p className="text-xs font-mono text-muted uppercase tracking-widest mb-3">
              Sections
            </p>

            <div className="space-y-2">
              <button
                onClick={() => setSection("chat")}
                className={`w-full h-10 px-3 rounded-xl flex items-center gap-2 text-sm font-medium transition-all
                  ${
                    section === "chat"
                      ? "bg-ink text-paper"
                      : "bg-paper text-muted hover:text-ink hover:bg-ash/60"
                  }`}
              >
                <MessageSquareText className="w-4 h-4" />
                Document Chat
              </button>

              <button
                onClick={() => setSection("quiz")}
                className={`w-full h-10 px-3 rounded-xl flex items-center gap-2 text-sm font-medium transition-all
                  ${
                    section === "quiz"
                      ? "bg-ink text-paper"
                      : "bg-paper text-muted hover:text-ink hover:bg-ash/60"
                  }`}
              >
                <FileQuestion className="w-4 h-4" />
                Quiz Generator
              </button>
            </div>
          </div>

          <div className="flex-1 min-h-0">
            <p className="text-xs font-mono text-muted uppercase tracking-widest mb-3">
              Documents
            </p>

            <UploadBox
              onUploaded={(msg, chunks) =>
                setUploadNote(`${chunks} chunks added`)
              }
            />

            {uploadNote && (
              <p className="mt-2 text-xs text-teal font-medium">
                {uploadNote}
              </p>
            )}
          </div>

          <div className="pt-4 border-t border-ash">
            <p className="text-xs text-muted leading-relaxed">
              Powered by <strong className="text-ink">Ollama</strong> ·{" "}
              <strong className="text-ink">Qdrant</strong> ·{" "}
              <strong className="text-ink">Whisper</strong>
            </p>
            <p className="text-xs text-muted mt-1">
              Arabic &amp; English support
            </p>
          </div>
        </div>
      </aside>

      <main className="flex flex-col flex-1 min-w-0 h-full">
        <header className="flex items-center gap-3 px-4 py-3 border-b border-ash bg-white flex-shrink-0">
          <button
            onClick={() => setSidebarOpen((o) => !o)}
            className="p-1.5 rounded-lg hover:bg-ash transition-colors"
            title="Toggle sidebar"
          >
            <svg
              className="w-5 h-5 text-muted"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.8}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5"
              />
            </svg>
          </button>

          <h1 className="text-sm font-semibold text-ink">
            {section === "chat" ? "Document Chat" : "Quiz Generator"}
          </h1>

          <div className="ml-auto text-xs text-muted font-mono hidden sm:block">
            llama3.2 · nomic-embed-text · Qdrant
          </div>
        </header>

        <div className="flex-1 min-h-0">
          {section === "chat" ? <ChatBox /> : <QuizGenerator />}
        </div>
      </main>
    </div>
  );
}