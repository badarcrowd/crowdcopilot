export function requireEnv(name: string) {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value;
}

export function appUrl() {
  return process.env.NEXT_PUBLIC_APP_URL ?? "http://localhost:3000";
}

export function defaultTenantId() {
  return process.env.DEFAULT_TENANT_ID ?? "default";
}

export function chatModelName() {
  return process.env.OPENAI_MODEL ?? "gpt-5";
}

export function smallModelName() {
  return process.env.OPENAI_SMALL_MODEL ?? "gpt-5-mini";
}

export function embeddingModelName() {
  return process.env.OPENAI_EMBEDDING_MODEL ?? "text-embedding-3-small";
}

export function transcriptionModelName() {
  return process.env.OPENAI_TRANSCRIPTION_MODEL ?? "gpt-4o-mini-transcribe";
}

export function ttsModelName() {
  return process.env.OPENAI_TTS_MODEL ?? "gpt-4o-mini-tts";
}

export function ttsVoiceName() {
  return process.env.OPENAI_TTS_VOICE ?? "marin";
}

export function ttsResponseFormat() {
  return process.env.OPENAI_TTS_FORMAT ?? "mp3";
}

export function realtimeModelName() {
  return process.env.OPENAI_REALTIME_MODEL ?? "gpt-realtime";
}

export function agentUseSmallModel() {
  return process.env.AGENT_USE_SMALL_MODEL !== "false";
}

export function openaiApiKey() {
  return requireEnv("OPENAI_API_KEY");
}
