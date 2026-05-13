import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import {
  openaiApiKey,
  realtimeModelName,
  transcriptionModelName,
  ttsVoiceName
} from "@/lib/env";
import { rateLimit } from "@/lib/security/rate-limit";

export const runtime = "nodejs";

const openaiClient = new OpenAI({
  apiKey: openaiApiKey()
});

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type"
};

const REALTIME_INSTRUCTIONS = `You are Crowd Agent — a premium, consultative voice copilot for Crowd Digital website visitors. Sound like a strategist, never a form or survey bot.

OPENING:
Lead with a warm, light, consultative ask for the three essentials in ONE natural sentence: website, work email, and best contact number with country code. Frame it around the value the visitor gets ("so I can run a quick analysis"). Never use comma-lists like "please provide…" or rigid form phrasing.
Good opener: "Happy to help. I can run a quick analysis for you — just send over your website, work email, and contact number with country code."

AS SOON AS THE WEBSITE IS SHARED:
Silently start the analysis in the background. Continue the conversation naturally — do not announce technical processing.

PROGRESSIVE DISCLOSURE — ask the rest casually, one at a time, after the opening:
"By the way, what industry are you in?" — for sector
"And which company are you working with?" — for company
"Which region should we route this through — Middle East, Europe, Asia, or the US?" — for office location
Then business challenge, success criteria, budget, start date, RFP details — one short question per turn.

SOFT VALIDATION — never say "invalid":
Phone missing country code → "That number looks incomplete — could you resend it with country code?"
Email unclear → "Hmm, that email doesn't look right — mind double-checking it?"
URL unclear → "That URL doesn't look right — try something like https://yoursite.com"

STYLE:
Speak in 1 or 2 short sentences. No markdown, no lists, no robotic onboarding phrases. If a visitor volunteers several fields at once, acknowledge what was captured and continue from the earliest missing field — never re-ask. If they ask about pricing, map them to starter, growth, or enterprise support and invite a strategy call when intent is strong.`;

function safeVoice(value: unknown) {
  if (typeof value !== "string") return ttsVoiceName();
  return /^[a-z0-9_-]{2,64}$/i.test(value) ? value : ttsVoiceName();
}

export async function OPTIONS() {
  return new Response(null, {
    status: 204,
    headers: CORS_HEADERS
  });
}

export async function POST(req: NextRequest) {
  const limited = await rateLimit(req, "voice-realtime-session", 20, 60);
  if (limited) return limited;

  try {
    const body = await req.json().catch(() => ({}));
    const voice = safeVoice(body?.voice);

    const session = await openaiClient.realtime.clientSecrets.create({
      expires_after: {
        anchor: "created_at",
        seconds: 120
      },
      session: {
        type: "realtime",
        model: realtimeModelName(),
        output_modalities: ["audio"],
        instructions: REALTIME_INSTRUCTIONS,
        max_output_tokens: 220,
        audio: {
          input: {
            noise_reduction: { type: "near_field" },
            transcription: {
              model: transcriptionModelName(),
              language: "en",
              prompt:
                "Crowd Digital lead conversation: SEO, PPC, CRO, WordPress, website audit, RFP, UAE, Europe, China, budget."
            },
            turn_detection: {
              type: "server_vad",
              create_response: true,
              interrupt_response: true,
              prefix_padding_ms: 250,
              silence_duration_ms: 420,
              threshold: 0.5,
              idle_timeout_ms: 6000
            }
          },
          output: {
            voice,
            speed: 1.04
          }
        }
      }
    });

    return NextResponse.json(
      {
        clientSecret: session.value,
        expiresAt: session.expires_at,
        model: realtimeModelName(),
        session: session.session
      },
      { headers: CORS_HEADERS }
    );
  } catch (error) {
    console.error("[Realtime Session Error]", error);
    return NextResponse.json(
      { error: "Failed to create realtime voice session" },
      { status: 500, headers: CORS_HEADERS }
    );
  }
}
