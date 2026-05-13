"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import ReactMarkdown from "react-markdown";
import {
  Bot,
  CheckCircle2,
  Loader2,
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";

import { AuditCard } from "@/components/agent/audit-card";
import { BudgetCard, LeadScoreCard } from "@/components/agent/insight-card";
import { BookingPanel } from "@/components/agent/booking-panel";
import type { LeadFormData } from "@/lib/agent/lead-score";
import type { AuditResult } from "@/lib/agent/pagespeed";
import type { CalendarSlot } from "@/lib/agent/calendar";
import {
  getVoiceRecorderOptions,
  voiceCaptureConstraints,
} from "@/lib/voice/browser";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ToolInvocation {
  toolCallId: string;
  toolName: string;
  state: "partial-call" | "call" | "result";
  args: Record<string, unknown>;
  result?: Record<string, unknown>;
}

type Phase = "greeting" | "discovery" | "audit" | "qualification" | "booking";

const FIELD_LABELS: Partial<Record<keyof LeadFormData, string>> = {
  name: "Name",
  email: "Email",
  company: "Company",
  website: "Website",
  sector: "Industry",
  challenge: "Challenge",
  budget: "Budget",
};

const GREETING_TEXT =
  "Happy to help. I can run a quick analysis for you — just send over your website, work email, and contact number with country code, and I'll take it from there.";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function createSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

function readPartText(message: {
  parts?: Array<{ type: string; text?: string }>;
}): string {
  return (
    message.parts
      ?.filter((p) => p.type === "text")
      .map((p) => p.text ?? "")
      .join("") ?? ""
  );
}

function getToolInvocations(message: {
  parts?: Array<{ type: string; [k: string]: unknown }>;
}): ToolInvocation[] {
  return (
    (message.parts?.filter(
      (p) => p.type === "tool-invocation"
    ) as unknown as Array<{ toolInvocation: ToolInvocation }>) ?? []
  ).map((p) => p.toolInvocation);
}

function stripMarkdown(text: string): string {
  return text
    .replace(/[*_`#>[\]]/g, "")
    .replace(/\n+/g, " ")
    .trim()
    .slice(0, 500);
}

// ---------------------------------------------------------------------------
// TTS hook
// ---------------------------------------------------------------------------

function useTTS() {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [muted, setMuted] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const mutedRef = useRef(false);

  function stopAudio() {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    setIsSpeaking(false);
  }

  async function speak(text: string) {
    if (mutedRef.current || !text.trim()) return;
    stopAudio();
    const clean = stripMarkdown(text);
    if (!clean) return;

    setIsSpeaking(true);
    try {
      const res = await fetch("/api/voice/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: clean }),
      });
      if (!res.ok) throw new Error("TTS failed");

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audioRef.current = audio;

      audio.onended = () => {
        URL.revokeObjectURL(url);
        setIsSpeaking(false);
        audioRef.current = null;
      };
      audio.onerror = () => {
        URL.revokeObjectURL(url);
        setIsSpeaking(false);
        audioRef.current = null;
      };

      await audio.play();
    } catch {
      setIsSpeaking(false);
    }
  }

  function toggleMute() {
    mutedRef.current = !mutedRef.current;
    setMuted(mutedRef.current);
    if (mutedRef.current) stopAudio();
  }

  return { speak, stopAudio, isSpeaking, muted, toggleMute };
}

// ---------------------------------------------------------------------------
// Voice recorder hook
// ---------------------------------------------------------------------------

function useVoiceRecorder(onTranscript: (text: string) => void) {
  const [recStatus, setRecStatus] = useState<
    "idle" | "recording" | "processing"
  >("idle");
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  async function start() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia(
        voiceCaptureConstraints
      );
      streamRef.current = stream;
      chunksRef.current = [];

      const recorder = new MediaRecorder(stream, getVoiceRecorderOptions());

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        streamRef.current?.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType });
        setRecStatus("processing");
        try {
          const fd = new FormData();
          fd.append("audio", blob, "audio.webm");
          const res = await fetch("/api/voice/transcribe", {
            method: "POST",
            body: fd,
          });
          if (!res.ok) throw new Error("Transcription failed");
          const { text } = await res.json();
          if (text?.trim()) onTranscript(text.trim());
        } catch {
          // Silently ignore
        } finally {
          setRecStatus("idle");
        }
      };

      mediaRecorderRef.current = recorder;
      recorder.start(250);
      setRecStatus("recording");
    } catch {
      setRecStatus("idle");
    }
  }

  function stop() {
    mediaRecorderRef.current?.stop();
  }

  function toggle() {
    if (recStatus === "idle") start();
    else if (recStatus === "recording") stop();
  }

  return { recStatus, toggle };
}

// ---------------------------------------------------------------------------
// Main CrowdAgent — Floating Voice Widget
// ---------------------------------------------------------------------------

export function CrowdAgent() {
  const [isOpen, setIsOpen] = useState(false);
  const [hasGreeted, setHasGreeted] = useState(false);
  const [sessionId] = useState(createSessionId);
  const [textInput, setTextInput] = useState("");
  const [form, setForm] = useState<LeadFormData>({});
  const [phase, setPhase] = useState<Phase>("greeting");
  const [slots, setSlots] = useState<CalendarSlot[]>([]);
  const [showBooking, setShowBooking] = useState(false);
  const lastSpokenIdRef = useRef<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { speak, stopAudio, isSpeaking, muted, toggleMute } = useTTS();

  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: "/api/agent/chat",
        body: () => ({ sessionId, tenantId: "default", leadData: form }),
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [sessionId]
  );

  const { messages: chatMessages, sendMessage, status } = useChat({
    id: sessionId,
    transport,
    messages: [],
  });

  const GREETING_MSG = useMemo(
    () => ({
      id: "greeting-static",
      role: "assistant" as const,
      parts: [{ type: "text" as const, text: GREETING_TEXT }],
      metadata: {} as Record<string, unknown>,
    }),
    []
  );

  const messages = chatMessages.length === 0 ? [GREETING_MSG] : chatMessages;
  const isBusy = status === "submitted" || status === "streaming";

  const handleTranscript = useCallback(
    (text: string) => {
      stopAudio();
      sendMessage({ parts: [{ type: "text", text }] });
    },
    [sendMessage, stopAudio]
  );

  const { recStatus, toggle: toggleRecording } =
    useVoiceRecorder(handleTranscript);

  // Greet with TTS on first open
  useEffect(() => {
    if (isOpen && !hasGreeted) {
      setHasGreeted(true);
      speak(GREETING_TEXT);
    }
  }, [isOpen, hasGreeted, speak]);

  // Auto-speak new assistant messages after streaming completes
  useEffect(() => {
    if (status === "submitted" || status === "streaming") return;
    const last = [...messages].reverse().find((m) => m.role === "assistant");
    if (!last || last.id === lastSpokenIdRef.current) return;
    if (last.id === "greeting-static") return;

    const text = readPartText(last as Parameters<typeof readPartText>[0]);
    if (!text) return;

    lastSpokenIdRef.current = last.id;
    speak(text);
  }, [status, messages, speak]);

  // Parse tool results to update UI
  useEffect(() => {
    for (const message of messages) {
      if (message.role !== "assistant") continue;
      const invocations = getToolInvocations(
        message as Parameters<typeof getToolInvocations>[0]
      );
      for (const inv of invocations) {
        if (inv.state !== "result" || !inv.result) continue;

        if (inv.toolName === "extract_lead_data" && inv.result.extracted) {
          const extracted = inv.result.extracted as Partial<LeadFormData>;
          setForm((prev) => ({
            ...prev,
            ...Object.fromEntries(
              Object.entries(extracted).filter(
                ([, v]) => typeof v === "string" && v
              )
            ),
          }));
        }

        if (inv.toolName === "run_website_audit") setPhase("audit");
        if (inv.toolName === "validate_budget") setPhase("qualification");
        if (inv.toolName === "score_lead") setPhase("qualification");

        if (inv.toolName === "get_calendar_slots") {
          const rawSlots = inv.result.slots as CalendarSlot[] | undefined;
          if (rawSlots?.length) {
            setSlots(rawSlots);
            setShowBooking(true);
            setPhase("booking");
          }
        }
      }
    }
  }, [messages]);

  // Advance phase from greeting on first user message
  useEffect(() => {
    if (
      messages.filter((m) => m.role === "user").length >= 1 &&
      phase === "greeting"
    ) {
      setPhase("discovery");
    }
  }, [messages, phase]);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function handleTextSend() {
    const text = textInput.trim();
    if (!text || isBusy) return;
    stopAudio();
    setTextInput("");
    sendMessage({ parts: [{ type: "text", text }] });
  }

  const collectedFields = (
    Object.entries(form) as [keyof LeadFormData, string][]
  ).filter(([, v]) => v);

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end gap-3">
      {/* Floating panel */}
      {isOpen && (
        <div
          className="w-[380px] bg-white rounded-3xl shadow-2xl border border-neutral-200 flex flex-col overflow-hidden"
          style={{ height: "580px" }}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-indigo-600 to-violet-600 text-white shrink-0">
            <div className="flex items-center gap-2.5">
              <div className="relative">
                <div className="h-8 w-8 rounded-full bg-white/20 flex items-center justify-center">
                  <Bot className="h-4 w-4" />
                </div>
                <span
                  className={`absolute -bottom-0.5 -right-0.5 h-2.5 w-2.5 rounded-full border-2 border-indigo-600 ${
                    isBusy || isSpeaking
                      ? "bg-amber-400 animate-pulse"
                      : "bg-emerald-400"
                  }`}
                />
              </div>
              <div>
                <p className="text-sm font-semibold leading-none">
                  Crowd Agent
                </p>
                <p className="text-[10px] text-indigo-200 mt-0.5">
                  {recStatus === "recording"
                    ? "Listening…"
                    : recStatus === "processing"
                    ? "Processing voice…"
                    : isSpeaking
                    ? "Speaking…"
                    : isBusy
                    ? "Thinking…"
                    : "Voice consultant · Online"}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={toggleMute}
                title={muted ? "Unmute voice" : "Mute voice"}
                className="p-1.5 rounded-full hover:bg-white/20 transition-colors"
              >
                {muted ? (
                  <VolumeX className="h-4 w-4" />
                ) : (
                  <Volume2 className="h-4 w-4" />
                )}
              </button>
              <button
                type="button"
                onClick={() => setIsOpen(false)}
                className="p-1.5 rounded-full hover:bg-white/20 transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3 min-h-0">
            {messages.map((message) => {
              const text = readPartText(
                message as Parameters<typeof readPartText>[0]
              );
              const invocations = getToolInvocations(
                message as Parameters<typeof getToolInvocations>[0]
              );
              const isUser = message.role === "user";

              return (
                <div
                  key={message.id}
                  className={`flex gap-2 ${isUser ? "flex-row-reverse" : "flex-row"}`}
                >
                  {!isUser && (
                    <div className="h-6 w-6 rounded-full bg-indigo-600 flex items-center justify-center shrink-0 mt-1">
                      <Bot className="h-3 w-3 text-white" />
                    </div>
                  )}

                  <div
                    className={`max-w-[85%] space-y-2 flex flex-col ${
                      isUser ? "items-end" : "items-start"
                    }`}
                  >
                    {text && (
                      <div
                        className={`rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed ${
                          isUser
                            ? "bg-indigo-600 text-white rounded-tr-sm"
                            : "bg-neutral-100 text-neutral-800 rounded-tl-sm"
                        }`}
                      >
                        {isUser ? (
                          text
                        ) : (
                          <ReactMarkdown>{text}</ReactMarkdown>
                        )}
                      </div>
                    )}

                    {/* Tool result cards */}
                    {invocations.map((inv) => {
                      if (inv.state !== "result" || !inv.result) return null;

                      if (
                        inv.toolName === "run_website_audit" &&
                        inv.result.success
                      ) {
                        return (
                          <div key={inv.toolCallId} className="w-full">
                            <AuditCard
                              audit={inv.result as unknown as AuditResult}
                            />
                          </div>
                        );
                      }
                      if (inv.toolName === "validate_budget") {
                        return (
                          <div key={inv.toolCallId} className="w-full">
                            <BudgetCard
                              budget={
                                inv.result as Parameters<
                                  typeof BudgetCard
                                >[0]["budget"]
                              }
                            />
                          </div>
                        );
                      }
                      if (inv.toolName === "score_lead") {
                        return (
                          <div key={inv.toolCallId} className="w-full">
                            <LeadScoreCard
                              score={
                                inv.result as Parameters<
                                  typeof LeadScoreCard
                                >[0]["score"]
                              }
                            />
                          </div>
                        );
                      }
                      return null;
                    })}

                    {/* Typing indicator */}
                    {!isUser &&
                      message === messages[messages.length - 1] &&
                      isBusy &&
                      !text && (
                        <div className="flex gap-1 px-4 py-3 bg-neutral-100 rounded-2xl rounded-tl-sm">
                          {[0, 1, 2].map((i) => (
                            <div
                              key={i}
                              className="h-1.5 w-1.5 rounded-full bg-neutral-400 animate-bounce"
                              style={{ animationDelay: `${i * 0.15}s` }}
                            />
                          ))}
                        </div>
                      )}
                  </div>
                </div>
              );
            })}

            {/* Booking panel inline */}
            {showBooking && slots.length > 0 && (
              <div className="mt-2">
                <BookingPanel
                  slots={slots}
                  prefillName={form.name}
                  prefillEmail={form.email}
                  onBooked={(slot) => {
                    setShowBooking(false);
                    sendMessage({
                      parts: [
                        {
                          type: "text",
                          text: `I've booked the slot on ${slot.displayDate} at ${slot.displayTime}.`,
                        },
                      ],
                    });
                  }}
                />
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Collected fields chips */}
          {collectedFields.length > 0 && (
            <div className="px-4 py-2 border-t border-neutral-100 flex flex-wrap gap-1.5 shrink-0">
              {collectedFields.map(([key]) => (
                <span
                  key={key}
                  className="inline-flex items-center gap-1 text-[10px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-full px-2 py-0.5"
                >
                  <CheckCircle2 className="h-2.5 w-2.5" />
                  {FIELD_LABELS[key] ?? key}
                </span>
              ))}
            </div>
          )}

          {/* Voice + text input */}
          <div className="px-4 pb-4 pt-3 border-t border-neutral-100 space-y-3 shrink-0">
            <div className="flex flex-col items-center gap-1.5">
              <button
                type="button"
                onClick={() => {
                  if (recStatus === "idle") stopAudio();
                  toggleRecording();
                }}
                disabled={isBusy || recStatus === "processing"}
                className={`relative h-16 w-16 rounded-full flex items-center justify-center transition-all focus:outline-none focus-visible:ring-4 focus-visible:ring-indigo-300 disabled:opacity-50 ${
                  recStatus === "recording"
                    ? "bg-red-500 scale-110"
                    : recStatus === "processing"
                    ? "bg-neutral-300"
                    : isSpeaking
                    ? "bg-violet-500"
                    : "bg-indigo-600 hover:bg-indigo-700 active:scale-95"
                }`}
              >
                {recStatus === "recording" && (
                  <>
                    <span className="absolute inset-0 rounded-full bg-red-400 animate-ping opacity-40 pointer-events-none" />
                    <span className="absolute -inset-2 rounded-full border-2 border-red-300 animate-pulse pointer-events-none" />
                  </>
                )}
                {recStatus === "idle" && !isBusy && !isSpeaking && (
                  <span className="absolute inset-0 rounded-full bg-indigo-400 animate-pulse opacity-30 pointer-events-none" />
                )}
                {isSpeaking && (
                  <span className="absolute inset-0 rounded-full bg-violet-400 animate-pulse opacity-40 pointer-events-none" />
                )}

                {recStatus === "processing" ? (
                  <Loader2 className="h-7 w-7 text-white animate-spin" />
                ) : recStatus === "recording" ? (
                  <MicOff className="h-7 w-7 text-white" />
                ) : (
                  <Mic className="h-7 w-7 text-white" />
                )}
              </button>

              <p className="text-[11px] text-neutral-400">
                {recStatus === "recording"
                  ? "Tap to stop recording"
                  : recStatus === "processing"
                  ? "Processing…"
                  : isSpeaking
                  ? "Speaking — tap mic to interrupt"
                  : "Tap mic to speak"}
              </p>
            </div>

            {/* Text fallback */}
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleTextSend();
                }}
                placeholder="or type here…"
                className="flex-1 text-sm rounded-xl border border-neutral-200 px-3 py-2 focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-100"
              />
              <button
                type="button"
                onClick={handleTextSend}
                disabled={!textInput.trim() || isBusy}
                className="h-9 w-9 flex items-center justify-center rounded-xl bg-indigo-600 text-white disabled:opacity-40 hover:bg-indigo-700 transition-colors shrink-0"
              >
                {isBusy ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <svg
                    className="h-4 w-4"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={2}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                    />
                  </svg>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Floating trigger button */}
      <button
        type="button"
        onClick={() => setIsOpen((v) => !v)}
        title={isOpen ? "Close agent" : "Talk to Crowd Agent"}
        className={`relative h-16 w-16 rounded-full shadow-xl flex items-center justify-center transition-all hover:scale-105 active:scale-95 focus:outline-none focus-visible:ring-4 focus-visible:ring-indigo-300 ${
          isOpen
            ? "bg-neutral-700 hover:bg-neutral-800"
            : "bg-gradient-to-br from-indigo-600 to-violet-600"
        }`}
      >
        {!isOpen && (
          <span className="absolute inset-0 rounded-full bg-indigo-500 animate-ping opacity-25 pointer-events-none" />
        )}
        {isOpen ? (
          <X className="h-6 w-6 text-white" />
        ) : (
          <Mic className="h-7 w-7 text-white" />
        )}
      </button>
    </div>
  );
}
