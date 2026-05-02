"use client";

import { useEffect, useState } from "react";
import ChatBox from "@/components/ChatBox";
import UploadBox from "@/components/UploadBox";
import { checkHealth } from "@/services/api";

export default function Home() {
  const [healthy,    setHealthy]    = useState<boolean | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [uploadNote, setUploadNote]  = useState<string>("");

  useEffect(() => {
    checkHealth().then(setHealthy);
  }, []);

  return (
    <div className="flex h-screen overflow-hidden bg-paper">

      {/* ── Sidebar ─────────────────────────────────────────────────────── */}
      <aside
        className={`
          flex flex-col border-r border-ash bg-white
          transition-all duration-300 ease-in-out flex-shrink-0
          ${sidebarOpen ? "w-72" : "w-0 overflow-hidden"}
        `}
      >
        <div className="flex flex-col h-full p-5 gap-6 min-w-[18rem]">
          {/* Logo */}
          <div className="flex items-center gap-2.5 pt-1">
            <div className="w-8 h-8 rounded-lg bg-ink flex items-center justify-center">
              <svg className="w-4 h-4 text-paper" fill="none" viewBox="0 0 24 24"
                stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125
                    1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0
                    12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125
                    1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0
                    1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"/>
              </svg>
            </div>
            <span className="font-semibold text-ink">DocAssist AI</span>
          </div>

          {/* Backend status */}
          <div className="flex items-center gap-2 text-xs">
            <span className={`w-2 h-2 rounded-full ${
              healthy === null  ? "bg-muted animate-pulse" :
              healthy           ? "bg-teal" : "bg-accent"
            }`} />
            <span className="text-muted">
              {healthy === null ? "Connecting…" : healthy ? "Backend online" : "Backend offline"}
            </span>
          </div>

          {/* Upload */}
          <div className="flex-1 min-h-0">
            <p className="text-xs font-mono text-muted uppercase tracking-widest mb-3">
              Documents
            </p>
            <UploadBox
              onUploaded={(msg, chunks) =>
                setUploadNote(`✓ ${chunks} chunks added`)
              }
            />
            {uploadNote && (
              <p className="mt-2 text-xs text-teal font-medium">{uploadNote}</p>
            )}
          </div>

          {/* Footer */}
          <div className="pt-4 border-t border-ash">
            <p className="text-xs text-muted leading-relaxed">
              Powered by <strong className="text-ink">Ollama</strong> ·{" "}
              <strong className="text-ink">Qdrant</strong> ·{" "}
              <strong className="text-ink">Whisper</strong>
            </p>
            <p className="text-xs text-muted mt-1">Arabic &amp; English support</p>
          </div>
        </div>
      </aside>

      {/* ── Main ────────────────────────────────────────────────────────── */}
      <main className="flex flex-col flex-1 min-w-0 h-full">
        {/* Topbar */}
        <header className="flex items-center gap-3 px-4 py-3 border-b border-ash bg-white flex-shrink-0">
          <button
            onClick={() => setSidebarOpen((o) => !o)}
            className="p-1.5 rounded-lg hover:bg-ash transition-colors"
            title="Toggle sidebar"
          >
            <svg className="w-5 h-5 text-muted" fill="none" viewBox="0 0 24 24"
              stroke="currentColor" strokeWidth={1.8}>
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
            </svg>
          </button>
          <h1 className="text-sm font-semibold text-ink">Document Chat</h1>
          <div className="ml-auto text-xs text-muted font-mono hidden sm:block">
            llama3.2 · nomic-embed-text · Qdrant
          </div>
        </header>

        {/* Chat */}
        <div className="flex-1 min-h-0">
          <ChatBox />
        </div>
      </main>
    </div>
  );
}
