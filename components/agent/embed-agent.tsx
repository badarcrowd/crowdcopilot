"use client";

/**
 * EmbedAgent — Crowd Agent voice widget designed to run inside an iframe
 * on the Crowd WordPress site (thisiscrowd.local/contact-us/).
 *
 * Communication flow:
 * 1. Agent collects lead data via voice/text conversation
 * 2. On extract_lead_data → sends postMessage to parent window with field values
 * 3. Bridge script on WP page fills CF7 form fields + advances multi-step
 * 4. On booking → sends postMessage CROWD_AGENT_BOOK_MEETING
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import ReactMarkdown from "react-markdown";
import { Bot, CheckCircle2, Loader2, Mic, MicOff, Volume2, VolumeX } from "lucide-react";

import { AuditCard } from "@/components/agent/audit-card";
import { BudgetCard, LeadScoreCard } from "@/components/agent/insight-card";
import type { LeadFormData } from "@/lib/agent/lead-score";
import type { AuditResult } from "@/lib/agent/pagespeed";
import type { CalendarSlot } from "@/lib/agent/calendar";
import {
  getVoiceRecorderOptions,
  voiceCaptureConstraints,
} from "@/lib/voice/browser";

// ── Types ────────────────────────────────────────────────────────────────────

interface ToolInvocation {
  toolCallId: string;
  toolName: string;
  state: "partial-call" | "call" | "result";
  args: Record<string, unknown>;
  result?: Record<string, unknown>;
}

// Maps agent extract_lead_data keys → CF7 input[name] values used in crowd2024 theme
// The server now returns pre-mapped cf7Fields in the tool result — this is the fallback client-side map
const CF7_FIELD_MAP: Record<string, string> = {
  first_name: "first_name",
  last_name:  "last_name",
  email:      "email",
  phone:      "phone",
  company:    "company",
  website:    "website",
  sector:     "sector",
  challenge:  "business",  // CF7 field name is "business"
  budget:     "cost",      // CF7 field name is "cost"
  location:   "location",
  rfp:        "rfp",
};

const GREETING_TEXT =
  "Happy to help. I can run a quick analysis for you — just send over your website, work email, and contact number with country code, and I'll take it from there.";

// ── Helpers ──────────────────────────────────────────────────────────────────

function createSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = Math.trunc(Math.random() * 16);
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

function readPartText(message: { parts?: Array<{ type: string; text?: string }> }): string {
  return message.parts?.filter((p) => p.type === "text").map((p) => p.text ?? "").join("") ?? "";
}

function getToolInvocations(message: { parts?: Array<{ type: string; [k: string]: unknown }> }): ToolInvocation[] {
  return (
    (message.parts?.filter((p) => p.type === "tool-invocation") as unknown as Array<{ toolInvocation: ToolInvocation }>) ?? []
  ).map((p) => p.toolInvocation);
}

function stripMarkdown(text: string): string {
  return text.replaceAll(/[*_`#>[\]]/g, "").replaceAll(/\n+/g, " ").trim().slice(0, 500);
}

/** Send a postMessage to the parent WordPress page with CF7 field data */
function notifyParent(type: string, payload: Record<string, unknown>) {
  if (typeof window === "undefined" || window === window.parent) return;
  window.parent.postMessage({ source: "crowd-agent", type, payload }, "*");
}

// ── TTS hook ─────────────────────────────────────────────────────────────────

function useTTS(onSpeakingEnded?: () => void) {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [muted, setMuted] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const mutedRef = useRef(false);

  function stopAudio() {
    if (audioRef.current) { audioRef.current.pause(); audioRef.current = null; }
    setIsSpeaking(false);
  }

  async function speak(text: string) {
    console.log("[Agent TTS] Speak called with:", text?.slice(0, 30) + "...");
    if (mutedRef.current || !text.trim()) {
      console.log("[Agent TTS] Skipping: muted or empty text");
      return;
    }
    stopAudio();
    const clean = stripMarkdown(text);
    if (!clean) {
      console.warn("[Agent TTS] No text after markdown stripping");
      return;
    }
    console.log("[Agent TTS] Clean text:", clean?.slice(0, 30) + "...");
    setIsSpeaking(true);
    try {
      console.log("[Agent TTS] Requesting audio from API...");
      const res = await fetch("/api/voice/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: clean }),
      });
      if (!res.ok) {
        console.error("[Agent TTS] API error:", res.status, res.statusText);
        throw new Error("TTS failed");
      }
      const blob = await res.blob();
      console.log("[Agent TTS] Received audio blob:", blob.size, "bytes");
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audioRef.current = audio;
      audio.onended = () => {
        console.log("[Agent TTS] Audio finished playing");
        URL.revokeObjectURL(url);
        setIsSpeaking(false);
        audioRef.current = null;
        // Call the speaking-ended callback
        onSpeakingEnded?.();
      };
      audio.onerror = (e) => {
        console.error("[Agent TTS] Audio error:", e);
        URL.revokeObjectURL(url);
        setIsSpeaking(false);
        audioRef.current = null;
        // Still call the callback on error so listening can resume
        onSpeakingEnded?.();
      };
      console.log("[Agent TTS] Playing audio...");
      await audio.play();
    } catch (error) {
      console.error("[Agent TTS] Error:", error);
      setIsSpeaking(false);
      // Call callback on fetch error too
      onSpeakingEnded?.();
    }
  }

  function toggleMute() {
    mutedRef.current = !mutedRef.current;
    setMuted(mutedRef.current);
    if (mutedRef.current) stopAudio();
  }

  return { speak, stopAudio, isSpeaking, muted, toggleMute };
}

// ── Voice recorder hook ───────────────────────────────────────────────────────

interface VoiceRecorderOptions {
  silenceMs?: number;
  silenceThreshold?: number;
}

function useVoiceRecorder(onTranscript: (t: string) => void, options?: VoiceRecorderOptions) {
  const [recStatus, setRecStatus] = useState<"idle" | "recording" | "processing">("idle");
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const rafRef = useRef<number | null>(null);
  const silenceStartRef = useRef<number | null>(null);

  const silenceMs = options?.silenceMs ?? 1500;
  const silenceThreshold = options?.silenceThreshold ?? 8;

  async function start() {
    try {
      console.log("[Agent Voice] Starting recording with VAD...");
      const stream = await navigator.mediaDevices.getUserMedia(
        voiceCaptureConstraints
      );
      streamRef.current = stream;
      chunksRef.current = [];
      const recorder = new MediaRecorder(stream, getVoiceRecorderOptions());
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
          console.log("[Agent Voice] Audio chunk received:", e.data.size, "bytes");
        }
      };
      recorder.onstop = async () => {
        // Clean up VAD resources
        if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
        if (audioContextRef.current) await audioContextRef.current.close();
        audioContextRef.current = null;
        silenceStartRef.current = null;

        streamRef.current?.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType });
        console.log("[Agent Voice] Recording stopped. Total audio:", blob.size, "bytes");
        setRecStatus("processing");
        try {
          const fd = new FormData();
          fd.append("audio", blob, "audio.webm");
          console.log("[Agent Voice] Sending audio to transcribe endpoint...");
          const res = await fetch("/api/voice/transcribe", { method: "POST", body: fd });
          if (!res.ok) {
            console.error("[Agent Voice] Transcription API error:", res.status, res.statusText);
            throw new Error(`Transcription failed: ${res.status}`);
          }
          const { text } = await res.json();
          console.log("[Agent Voice] Transcription result:", text);
          if (text?.trim()) {
            console.log("[Agent Voice] Sending transcribed text to agent:", text);
            onTranscript(text.trim());
          } else {
            console.warn("[Agent Voice] No text content in transcription response");
          }
        } catch (error) {
          console.error("[Agent Voice] Transcription error:", error);
        } finally {
          setRecStatus("idle");
        }
      };
      recorder.onerror = (e) => {
        console.error("[Agent Voice] Recorder error:", e.error);
        setRecStatus("idle");
      };
      mediaRecorderRef.current = recorder;
      recorder.start(250);
      setRecStatus("recording");
      console.log("[Agent Voice] Recording started");

      // Start VAD analysis
      try {
        const ctx = new AudioContext();
        audioContextRef.current = ctx;
        const source = ctx.createMediaStreamSource(stream);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 512;
        source.connect(analyser);

        const buffer = new Uint8Array(analyser.fftSize);
        let silenceStart: number | null = null;

        function checkSilence() {
          analyser.getByteTimeDomainData(buffer);
          // Calculate RMS energy (values 0-255, centered at 128)
          const rms = Math.sqrt(
            buffer.reduce((s, v) => s + (v - 128) ** 2, 0) / buffer.length
          );

          if (rms < silenceThreshold) {
            if (silenceStart === null) {
              silenceStart = Date.now();
              console.log("[Agent Voice VAD] Silence detected, threshold:", silenceThreshold, "rms:", rms.toFixed(2));
            } else if (Date.now() - silenceStart > silenceMs) {
              console.log("[Agent Voice VAD] Silence duration exceeded, stopping recording");
              mediaRecorderRef.current?.stop();
              silenceStartRef.current = null;
              return;
            }
          } else {
            if (silenceStart !== null) {
              console.log("[Agent Voice VAD] Speech resumed, rms:", rms.toFixed(2));
            }
            silenceStart = null;
          }

          rafRef.current = requestAnimationFrame(checkSilence);
        }

        rafRef.current = requestAnimationFrame(checkSilence);
      } catch (error) {
        console.warn("[Agent Voice VAD] Could not setup VAD, falling back to manual stop:", error);
      }
    } catch (error) {
      console.error("[Agent Voice] Failed to start recording:", error);
      setRecStatus("idle");
      alert("Microphone access denied. Please enable microphone permissions.");
    }
  }

  function stop() {
    if (recStatus === "recording") {
      console.log("[Agent Voice] Stopping recording...");
      mediaRecorderRef.current?.stop();
    }
  }

  function toggle() {
    if (recStatus === "idle") start();
    else if (recStatus === "recording") stop();
  }

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      if (audioContextRef.current) audioContextRef.current.close();
    };
  }, []);

  return { recStatus, toggle, start, stop };
}

// ── EmbedAgent ────────────────────────────────────────────────────────────────

export function EmbedAgent() {
  const [sessionId] = useState(createSessionId);
  const [textInput, setTextInput] = useState("");
  const [form, setForm] = useState<LeadFormData>({});
  const [slots, setSlots] = useState<CalendarSlot[]>([]);
  const [conversationActive, setConversationActive] = useState(false);
  const conversationActiveRef = useRef(false);
  const lastSpokenIdRef = useRef<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const hasGreetedRef = useRef(false);

  // Keep ref in sync with state
  useEffect(() => {
    conversationActiveRef.current = conversationActive;
  }, [conversationActive]);

  const { speak, stopAudio, isSpeaking, muted, toggleMute } = useTTS(() => {
    // Bot finished speaking — auto-start mic if conversation is active
    console.log("[Agent Conversation] Bot finished speaking, conversationActive:", conversationActiveRef.current);
    if (conversationActiveRef.current && !muted) {
      console.log("[Agent Conversation] Auto-starting mic...");
      voiceRecorder.start();
    }
  });

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
      console.log("[Agent Chat] Received transcript:", text);
      stopAudio();
      console.log("[Agent Chat] Sending message to agent API...");
      sendMessage({ parts: [{ type: "text", text }] });
      console.log("[Agent Chat] Message sent");
    },
    [sendMessage, stopAudio]
  );

  const voiceRecorder = useVoiceRecorder(handleTranscript, {
    silenceMs: 1500,
    silenceThreshold: 8
  });
  const { recStatus, toggle: toggleRecording } = voiceRecorder;

  // Auto-start conversation after greeting (when user clicks "Start conversation")
  useEffect(() => {
    if (conversationActive && !isSpeaking && hasGreetedRef.current) {
      console.log("[Agent Greeting] Conversation active but not speaking, auto-starting mic...");
      voiceRecorder.start();
    }
  }, [conversationActive, isSpeaking]);

  // Auto-speak new assistant messages after streaming completes
  useEffect(() => {
    console.log("[Agent Chat] Status:", status, "Messages count:", messages.length);
    if (status === "submitted" || status === "streaming") {
      console.log("[Agent Chat] Still busy, waiting for response...");
      return;
    }
    const last = [...messages].reverse().find((m) => m.role === "assistant");
    if (!last) {
      console.log("[Agent Chat] No assistant message found");
      return;
    }
    if (last.id === lastSpokenIdRef.current) {
      console.log("[Agent Chat] Already spoke this message, skipping");
      return;
    }
    if (last.id === "greeting-static") {
      console.log("[Agent Chat] Skipping greeting message");
      return;
    }
    const text = readPartText(last as Parameters<typeof readPartText>[0]);
    console.log("[Agent Chat] New assistant message:", text?.slice(0, 50) + "...");
    if (!text) {
      console.warn("[Agent Chat] No text content in assistant message");
      return;
    }
    lastSpokenIdRef.current = last.id;
    console.log("[Agent Chat] Speaking response...");
    speak(text);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, messages]);

  // Parse tool results — update local form state + notify parent window
  useEffect(() => {
    for (const message of messages) {
      if (message.role !== "assistant") continue;
      const invocations = getToolInvocations(message as Parameters<typeof getToolInvocations>[0]);
      for (const inv of invocations) {
        if (inv.state !== "result" || !inv.result) continue;

        if (inv.toolName === "extract_lead_data" && inv.result.extracted) {
          const extracted = inv.result.extracted as Record<string, string>;
          const newFields = Object.fromEntries(
            Object.entries(extracted).filter(([, v]) => typeof v === "string" && v)
          ) as Partial<LeadFormData>;

          setForm((prev) => {
            const updated = { ...prev, ...newFields };

            // Prefer server-computed cf7Fields (already mapped), fall back to client map
            const cf7Fields: Record<string, string> =
              (inv.result!.cf7Fields as Record<string, string> | undefined) ??
              Object.fromEntries(
                Object.entries(CF7_FIELD_MAP)
                  .map(([agentKey, cf7Key]) => [cf7Key, updated[agentKey as keyof LeadFormData] as string])
                  .filter(([, v]) => v)
              );

            // Notify parent WP page to fill CF7 form
            if (Object.keys(cf7Fields).length > 0) {
              notifyParent("CROWD_AGENT_FILL_FORM", { fields: cf7Fields });
            }
            return updated;
          });
        }

        if (inv.toolName === "get_calendar_slots") {
          const rawSlots = inv.result.slots as CalendarSlot[] | undefined;
          if (rawSlots?.length) {
            setSlots(rawSlots);
            // Tell parent to advance to booking section
            notifyParent("CROWD_AGENT_SHOW_BOOKING", { slots: rawSlots });
          }
        }

        if (inv.toolName === "score_lead" && inv.result) {
          notifyParent("CROWD_AGENT_LEAD_SCORED", { score: inv.result });
        }
      }
    }
  }, [messages]);

  // Auto-scroll
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  function handleTextSend() {
    const text = textInput.trim();
    if (!text || isBusy) return;
    stopAudio();
    setTextInput("");
    sendMessage({ parts: [{ type: "text", text }] });
  }

  const collectedFields = (Object.entries(form) as [keyof LeadFormData, string][]).filter(([, v]) => v);

  function getStatusLabel() {
    if (recStatus === "recording") return "Listening…";
    if (recStatus === "processing") return "Processing voice…";
    if (isSpeaking) return "Speaking…";
    if (isBusy) return "Thinking…";
    return "Voice consultant · Online";
  }

  function getMicBg() {
    if (recStatus === "recording") return "bg-red-500 scale-110";
    if (recStatus === "processing") return "bg-neutral-300";
    if (isSpeaking) return "bg-violet-500";
    return "bg-indigo-600 hover:bg-indigo-700 active:scale-95";
  }

  function getMicHint() {
    if (recStatus === "recording") return "Tap to stop";
    if (recStatus === "processing") return "Processing…";
    if (isSpeaking) return "Speaking — tap to interrupt";
    return "Tap mic to speak";
  }

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div
      className="flex flex-col bg-white"
      style={{ width: "100vw", height: "100vh", fontFamily: "system-ui, sans-serif" }}
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
                isBusy || isSpeaking ? "bg-amber-400 animate-pulse" : "bg-emerald-400"
              }`}
            />
          </div>
          <div>
            <p className="text-sm font-semibold leading-none">Crowd Agent</p>
            <p className="text-[10px] text-indigo-200 mt-0.5">{getStatusLabel()}</p>
          </div>
        </div>
        <button
          type="button"
          onClick={toggleMute}
          title={muted ? "Unmute" : "Mute voice"}
          className="p-1.5 rounded-full hover:bg-white/20 transition-colors"
        >
          {muted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3 min-h-0">
        {messages.map((message) => {
          const text = readPartText(message as Parameters<typeof readPartText>[0]);
          const invocations = getToolInvocations(message as Parameters<typeof getToolInvocations>[0]);
          const isUser = message.role === "user";

          return (
            <div key={message.id} className={`flex gap-2 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
              {!isUser && (
                <div className="h-6 w-6 rounded-full bg-indigo-600 flex items-center justify-center shrink-0 mt-1">
                  <Bot className="h-3 w-3 text-white" />
                </div>
              )}
              <div className={`max-w-[85%] space-y-2 flex flex-col ${isUser ? "items-end" : "items-start"}`}>
                {text && (
                  <div
                    className={`rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed ${
                      isUser
                        ? "bg-indigo-600 text-white rounded-tr-sm"
                        : "bg-neutral-100 text-neutral-800 rounded-tl-sm"
                    }`}
                  >
                    {isUser ? text : <ReactMarkdown>{text}</ReactMarkdown>}
                  </div>
                )}

                {invocations.map((inv) => {
                  if (inv.state !== "result" || !inv.result) return null;
                  if (inv.toolName === "run_website_audit" && inv.result.success) {
                    return (
                      <div key={inv.toolCallId} className="w-full">
                        <AuditCard audit={inv.result as unknown as AuditResult} />
                      </div>
                    );
                  }
                  if (inv.toolName === "validate_budget") {
                    return (
                      <div key={inv.toolCallId} className="w-full">
                        <BudgetCard budget={inv.result as Parameters<typeof BudgetCard>[0]["budget"]} />
                      </div>
                    );
                  }
                  if (inv.toolName === "score_lead") {
                    return (
                      <div key={inv.toolCallId} className="w-full">
                        <LeadScoreCard score={inv.result as Parameters<typeof LeadScoreCard>[0]["score"]} />
                      </div>
                    );
                  }
                  return null;
                })}

                {!isUser && message === messages.at(-1) && isBusy && !text && (
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
              {key.charAt(0).toUpperCase() + key.slice(1)}
            </span>
          ))}
        </div>
      )}

      {/* Conversation control */}
      <div className="px-4 pt-3 border-t border-neutral-100 shrink-0">
        {!conversationActive ? (
          <button
            type="button"
            onClick={() => {
              console.log("[Agent UI] Starting conversation mode");
              hasGreetedRef.current = false;
              speak(GREETING_TEXT);
              setConversationActive(true);
            }}
            className="w-full px-4 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 text-white font-medium text-sm transition-all hover:shadow-lg active:scale-95"
          >
            Start Voice Conversation
          </button>
        ) : (
          <button
            type="button"
            onClick={() => {
              console.log("[Agent UI] Ending conversation mode");
              if (recStatus === "recording") voiceRecorder.stop();
              stopAudio();
              setConversationActive(false);
            }}
            className="w-full px-4 py-2.5 rounded-xl bg-neutral-300 text-neutral-700 font-medium text-sm transition-all hover:bg-neutral-400"
          >
            End Conversation
          </button>
        )}
      </div>

      {/* Voice + text input */}
      <div className="px-4 pb-4 pt-3 border-t border-neutral-100 space-y-3 shrink-0">
        <div className="flex flex-col items-center gap-1.5">
          <button
            type="button"
            onClick={() => { if (recStatus === "idle") stopAudio(); toggleRecording(); }}
            disabled={isBusy || recStatus === "processing" || conversationActive}
            title={conversationActive ? "Use voice conversation mode buttons above" : "Tap to record"}
            className={`relative h-16 w-16 rounded-full flex items-center justify-center transition-all focus:outline-none disabled:opacity-50 ${getMicBg()}`}
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
          <p className="text-[11px] text-neutral-400">{getMicHint()}</p>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="text"
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") handleTextSend(); }}
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
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
