"use client";

import { useState } from "react";
import { FileQuestion, Loader2, Sparkles } from "lucide-react";
import { generateQuiz } from "@/services/api";
import SourceBox from "./SourceBox";

type Language = "auto" | "ar" | "en";
type QuestionType = "mcq" | "true_false" | "short_answer" | "mixed";

const isArabicText = (text: string) => /[\u0600-\u06FF]/.test(text);

export default function QuizGenerator() {
    const [topic, setTopic] = useState("");
    const [numQuestions, setNumQuestions] = useState(5);
    const [questionType, setQuestionType] = useState<QuestionType>("mixed");
    const [language, setLanguage] = useState<Language>("auto");

    const [quiz, setQuiz] = useState("");
    const [sources, setSources] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const handleGenerate = async () => {
        if (loading) return;

        setLoading(true);
        setError("");
        setQuiz("");
        setSources("");

        try {
            const res = await generateQuiz({
                topic: topic.trim(),
                num_questions: numQuestions,
                question_type: questionType,
                language,
            });

            setQuiz(res.quiz);
            setSources(res.sources);
        } catch (err: any) {
            setError(err.message || "Quiz generation failed");
        } finally {
            setLoading(false);
        }
    };

    const quizIsAr = isArabicText(quiz);
    const topicIsAr = isArabicText(topic);

    return (
        <div className="flex flex-col h-full" dir="ltr">
            <div className="flex-1 overflow-y-auto px-4 py-6">
                <div className="max-w-4xl mx-auto space-y-5">
                    <div className="bg-white border border-ash rounded-2xl shadow-sm p-5">
                        <div className="flex items-center gap-3 mb-5">
                            <div className="w-10 h-10 rounded-xl bg-ink text-paper flex items-center justify-center">
                                <FileQuestion className="w-5 h-5" />
                            </div>
                            <div>
                                <h2 className="text-lg font-semibold text-ink">
                                    Quiz Generator
                                </h2>
                                <p className="text-sm text-muted">
                                    Generate questions from your uploaded documents
                                </p>
                            </div>
                        </div>

                        <div className="space-y-4">
                            <div>
                                <label className="block text-xs font-mono text-muted uppercase tracking-wider mb-2">
                                    Topic
                                </label>
                                <input
                                    value={topic}
                                    onChange={(e) => setTopic(e.target.value)}
                                    placeholder="Optional: NLP, history, transformers..."
                                    dir={topicIsAr ? "rtl" : "ltr"}
                                    className={`
                    w-full rounded-xl border border-ash bg-paper
                    px-4 py-3 text-[15px] text-ink outline-none
                    focus:border-teal focus:ring-2 focus:ring-teal/20
                    ${topicIsAr ? "text-right font-arabic" : "text-left"}
                  `}
                                />
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                                <div>
                                    <label className="block text-xs font-mono text-muted uppercase tracking-wider mb-2">
                                        Questions
                                    </label>
                                    <input
                                        type="number"
                                        min={1}
                                        max={20}
                                        value={numQuestions}
                                        onChange={(e) => setNumQuestions(Number(e.target.value))}
                                        className="w-full rounded-xl border border-ash bg-paper px-4 py-3 text-[15px] text-ink outline-none focus:border-teal focus:ring-2 focus:ring-teal/20"
                                    />
                                </div>

                                <div>
                                    <label className="block text-xs font-mono text-muted uppercase tracking-wider mb-2">
                                        Type
                                    </label>
                                    <select
                                        value={questionType}
                                        onChange={(e) =>
                                            setQuestionType(e.target.value as QuestionType)
                                        }
                                        className="w-full rounded-xl border border-ash bg-paper px-4 py-3 text-[15px] text-ink outline-none focus:border-teal focus:ring-2 focus:ring-teal/20"
                                    >
                                        <option value="mixed">Mixed</option>
                                        <option value="mcq">MCQ</option>
                                        <option value="true_false">True / False</option>
                                        <option value="short_answer">Short Answer</option>
                                    </select>
                                </div>

                                <div>
                                    <label className="block text-xs font-mono text-muted uppercase tracking-wider mb-2">
                                        Language
                                    </label>
                                    <select
                                        value={language}
                                        onChange={(e) => setLanguage(e.target.value as Language)}
                                        className="w-full rounded-xl border border-ash bg-paper px-4 py-3 text-[15px] text-ink outline-none focus:border-teal focus:ring-2 focus:ring-teal/20"
                                    >
                                        <option value="auto">Auto</option>
                                        <option value="ar">Arabic</option>
                                        <option value="en">English</option>
                                    </select>
                                </div>
                            </div>

                            <button
                                type="button"
                                onClick={handleGenerate}
                                disabled={loading}
                                className="h-12 px-5 rounded-xl bg-ink text-paper flex items-center gap-2 text-sm font-medium hover:bg-ink/80 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                            >
                                {loading ? (
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                ) : (
                                    <Sparkles className="w-4 h-4" />
                                )}
                                {loading ? "Generating..." : "Generate Quiz"}
                            </button>
                        </div>
                    </div>

                    {error && (
                        <div className="rounded-xl border border-accent/20 bg-accent/5 text-accent px-4 py-3 text-sm">
                            {error}
                        </div>
                    )}

                    {quiz && (
                        <div>
                            <div
                                className={`
                  bg-white border border-ash rounded-2xl shadow-sm
                  px-5 py-4 whitespace-pre-wrap text-[15px] leading-relaxed
                  ${quizIsAr ? "text-right font-arabic" : "text-left"}
                `}
                                dir={quizIsAr ? "rtl" : "ltr"}
                            >
                                {quiz}
                            </div>

                            {sources && <SourceBox sources={sources} />}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}