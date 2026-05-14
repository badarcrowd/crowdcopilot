import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import { createHash } from "crypto";
import type { SpeechCreateParams } from "openai/resources/audio/speech";
import {
  openaiApiKey,
  ttsModelName,
  ttsResponseFormat,
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

const AUDIO_MIME_TYPES = {
  mp3: "audio/mpeg",
  opus: "audio/ogg; codecs=opus",
  aac: "audio/aac",
  flac: "audio/flac",
  wav: "audio/wav",
  pcm: "audio/pcm"
} as const;

type SpeechFormat = keyof typeof AUDIO_MIME_TYPES;

const ttsCache = new Map<string, Buffer>();
const TTS_CACHE_LIMIT = 48;

function getSpeechFormat(): SpeechFormat {
  const format = ttsResponseFormat().toLowerCase();
  return format in AUDIO_MIME_TYPES ? (format as SpeechFormat) : "mp3";
}

function cleanSpeechText(text: string) {
  return text
    .replace(/<[^>]+>/g, " ")
    .replace(/[*_`#>[\]]/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 900);
}

function isCacheable(text: string) {
  return (
    text.length <= 260 &&
    !/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i.test(text) &&
    !/\+?\d[\d\s().-]{6,}/.test(text)
  );
}

function cacheKey(text: string, format: SpeechFormat) {
  return createHash("sha256")
    .update(`${ttsModelName()}|${ttsVoiceName()}|${format}|${text}`)
    .digest("hex");
}

function rememberAudio(key: string, buffer: Buffer) {
  while (ttsCache.size >= TTS_CACHE_LIMIT) {
    const oldestKey = ttsCache.keys().next().value;
    if (!oldestKey) break;
    ttsCache.delete(oldestKey);
  }
  ttsCache.set(key, buffer);
}

function bodyFromBuffer(buffer: Buffer): ArrayBuffer {
  const copy = new Uint8Array(buffer.byteLength);
  copy.set(buffer);
  return copy.buffer;
}

export async function OPTIONS() {
  return new Response(null, {
    status: 204,
    headers: CORS_HEADERS,
  });
}

export async function POST(req: NextRequest) {
  const limited = await rateLimit(req, "voice-speak", 60, 60);
  if (limited) return limited;

  try {
    const { text } = await req.json();
    const input = typeof text === "string" ? cleanSpeechText(text) : "";

    if (!input) {
      return NextResponse.json(
        { error: "No text provided" },
        { status: 400 }
      );
    }

    const model = ttsModelName();
    const format = getSpeechFormat();
    const key = cacheKey(input, format);
    const cached = isCacheable(input) ? ttsCache.get(key) : null;

    if (cached) {
      return new NextResponse(bodyFromBuffer(cached), {
        headers: {
          ...CORS_HEADERS,
          "Content-Type": AUDIO_MIME_TYPES[format],
          "Content-Length": cached.length.toString(),
          "Cache-Control": "no-store",
          "X-Voice-Cache": "hit"
        }
      });
    }

    const speechParams: SpeechCreateParams = {
      model,
      voice: ttsVoiceName(),
      input,
      speed: 1.04,
      response_format: format
    };

    if (!model.startsWith("tts-1")) {
      speechParams.instructions =
        "Speak warmly, clearly, and professionally. Keep the delivery concise and helpful for a website visitor.";
    }

    const audio = await openaiClient.audio.speech.create(speechParams);
    const buffer = Buffer.from(await audio.arrayBuffer());

    if (isCacheable(input)) {
      rememberAudio(key, buffer);
    }

    return new NextResponse(bodyFromBuffer(buffer), {
      headers: {
        ...CORS_HEADERS,
        "Content-Type": AUDIO_MIME_TYPES[format],
        "Content-Length": buffer.length.toString(),
        "Cache-Control": "no-store",
        "X-Voice-Cache": "miss"
      }
    });
  } catch (error) {
    console.error("[Voice Speech Error]", error);
    return NextResponse.json(
      { error: "Failed to generate speech" },
      { status: 500 }
    );
  }
}
