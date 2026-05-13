import { generateText, tool, stepCountIs } from "ai";
import { NextResponse } from "next/server";
import { z } from "zod";
import type { OpenAIResponsesProviderOptions } from "@ai-sdk/openai";
import { getChatModel } from "@/lib/ai/openai";
import { runPageSpeedAudit } from "@/lib/agent/pagespeed";
import { rateLimit } from "@/lib/security/rate-limit";

export async function OPTIONS() {
  return new Response(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    },
  });
}

export const runtime = "nodejs";
export const maxDuration = 60;

// ── FIELD VALIDATORS ──────────────────────────────────────────────────────
function validateField(name: string, value: string): string | null {
  const v = value.trim();
  if (name === "email") {
    return /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(v)
      ? null
      : "That doesn't look like a valid email address. Could you double-check it?";
  }
  if (name === "phone") {
    const digits = v.replaceAll(/\D/g, "");
    return digits.length >= 7
      ? null
      : "Phone number looks too short. Please include the country code (e.g. +971 50 123 4567).";
  }
  if (name === "website") {
    try {
      new URL(/^https?:\/\//i.test(v) ? v : `https://${v}`);
      return null;
    } catch {
      return "That URL doesn't look right. Try something like https://yoursite.com.";
    }
  }
  return null;
}

// ── SYSTEM PROMPT ─────────────────────────────────────────────────────────
// CRITICAL: only ONE tool call per turn, ALWAYS end with a text reply.
// Do NOT chain tools. Do NOT call score_lead, validate_budget, or any other
// tool — just extract_lead_data and run_website_audit.
const BOT_SYSTEM_PROMPT = `You are Aria — a warm, friendly AI guide for Crowd Digital, a premium growth agency. You sit as a floating bot on the contact page to help visitors fill the enquiry form naturally.

CRITICAL RULE: You may call AT MOST ONE tool per reply. After calling a tool, you MUST write a short conversational text response. Never end a turn with only a tool call.
VOICE-FIRST RULE: Replies may be spoken aloud. Use natural plain text only: no markdown, no emoji, no long lists, and no more than 2 short sentences.

YOUR GOAL: Sound like a premium strategist, never a form or survey bot. Collect the three essentials in ONE warm opener (website + work email + phone with country code), then progressively gather the rest casually as the conversation flows.

OPENING (first message of the conversation only):
Use a single warm, consultative line that asks for website + work email + phone with country code together — framed around the value the visitor gets ("so I can run a quick analysis"). Never use "please provide…", comma-separated field lists, or rigid form phrasing.
Good opener: "Happy to help. I can run a quick analysis for you — just send over your website, work email, and contact number with country code, and I'll take it from there."

WHEN THE WEBSITE IS CAPTURED:
Silently call run_website_audit and continue the conversation naturally. Never announce technical processing. Once the audit returns, drop ONE useful finding in conversational language before asking the next thing.

PROGRESSIVE FIELDS — ask casually, ONE per reply, after the opener:
1. first_name (only when it feels natural, e.g. before handoff — never demand it upfront)
2. company → "And which company are you working with?"
3. sector → "By the way, what industry are you in?" — options: Consumer Goods, Corporate & Business, Education, Entertainment, Health Beauty & Wellness, Real Estate, Retail, Sustainability, Technology, Travel & Tourism, Others
4. location (Crowd office) → "Which region should we route this through — Middle East, Europe, Asia, or the US?" — options: UAE, USA, Europe, China
5. business challenge → stored as "business"
6. success criteria → stored as "success"
7. budget → stored as "cost" — options: < $5,000 / $5k–$25k / $25k–$50k / $50k–$100k / +$100k
8. project start date → stored as "start"
9. RFP details → stored as "rfp"

CONVERSATION RULES:
- After the opener, ask ONE question per reply. Short, warm, consultative — 1–2 sentences max.
- Structure progressive replies as: [light acknowledgment] + [next casual question].
- If the user provides several fields in one message, call extract_lead_data ONCE with every field you can identify and continue from the earliest missing field. Never re-ask captured fields.
- Use the person's first name naturally once you have it, but do not block the flow to collect it.
- Never list all remaining fields or sound like a form, survey, or support ticket.
- When all fields are collected: "You're all set, [name]. The Crowd team will be in touch very soon."

SOFT VALIDATION — if extract_lead_data returns { ok: false, validationErrors }:
- Relay the issue gently and ask only for that single field again. Never say "invalid".
- Email unclear → "Hmm, that email doesn't look right — mind double-checking it?"
- Phone missing country code → "That number looks incomplete — could you resend it with country code?"
- URL unclear → "That URL doesn't look right — try something like https://yoursite.com"
- Never store or move on from an unclear field.

TONE — strategist, not support agent:
✗ "Please provide your email." → ✓ "What's the best email to reach you?"
✗ "Your budget has been noted." → ✓ "Got it — when are you hoping to start?"
✗ "Complete the following fields." → ✓ "Send over your website, work email, and best number, and I'll take it from there."`;

// ── REQUEST SCHEMA ─────────────────────────────────────────────────────────
const requestSchema = z.object({
  messages: z.array(
    z.object({
      role: z.enum(["user", "assistant"]),
      content: z.string()
    })
  ).min(1),
  leadData: z.record(z.string()).optional().default({}),
  sessionId: z.string().optional(),
  tenantId: z.string().optional()
});

const providerOptions: { openai: OpenAIResponsesProviderOptions } = {
  openai: {
    reasoningEffort: "low",
    textVerbosity: "low",
    promptCacheKey: "crowd-wordpress-form-agent",
    promptCacheRetention: "in_memory"
  }
};

// ── HANDLER ────────────────────────────────────────────────────────────────
export async function POST(request: Request) {
  const limited = await rateLimit(request, "bot-chat", 30, 60);
  if (limited) return limited;

  const json = await request.json().catch(() => null);
  if (!json) return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });

  const parsed = requestSchema.safeParse(json);
  if (!parsed.success) {
    return NextResponse.json({ error: "Invalid request", detail: parsed.error.flatten() }, { status: 400 });
  }

  const { messages, leadData } = parsed.data;
  const cf7Fields: Record<string, string> = {};
  let auditResult: Record<string, unknown> | null = null;

  // Field name mapping: AI name → CF7 field name
  const CF7_MAP: Record<string, string> = {
    first_name: "first_name",
    last_name:  "last_name",
    email:      "email",
    phone:      "phone",
    company:    "company",
    website:    "website",
    sector:     "sector",
    location:   "location",
    business:   "business",
    success:    "success",
    budget:     "cost",
    cost:       "cost",
    start:      "start",
    rfp:        "rfp"
  };

  try {
    const result = await generateText({
      model: getChatModel(true),
      system: BOT_SYSTEM_PROMPT,
      messages,
      stopWhen: stepCountIs(2),   // ONE tool call max → then text. Prevents chaining.
      maxRetries: 1,
      providerOptions,
      maxOutputTokens: 180,
      tools: {
        extract_lead_data: tool({
          description:
            "Store one or more field values the user just provided. " +
            "Call this once per turn with whatever fields the user mentioned.",
          inputSchema: z.object({
            first_name: z.string().optional(),
            last_name:  z.string().optional(),
            email:      z.string().optional(),
            phone:      z.string().optional(),
            company:    z.string().optional(),
            website:    z.string().optional(),
            sector:     z.string().optional(),
            location:   z.string().optional(),
            business:   z.string().optional(),
            success:    z.string().optional(),
            budget:     z.string().optional(),
            cost:       z.string().optional(),
            start:      z.string().optional(),
            rfp:        z.string().optional()
          }),
          execute: async (params) => {
            const errors: Record<string, string> = {};
            for (const [k, v] of Object.entries(params)) {
              if (!v || !CF7_MAP[k]) continue;
              const err = validateField(k, String(v));
              if (err) {
                errors[k] = err;
              } else {
                cf7Fields[CF7_MAP[k]] = String(v);
              }
            }
            return Object.keys(errors).length > 0
              ? { ok: false, validationErrors: errors }
              : { ok: true };
          }
        }),

        run_website_audit: tool({
          description:
            "Run a PageSpeed performance audit on the user's website. " +
            "Only call this when the user provides a URL.",
          inputSchema: z.object({
            url: z.string().describe("Full website URL to audit")
          }),
          execute: async ({ url }) => {
            const normalised = /^https?:\/\//i.test(url) ? url : `https://${url}`;
            cf7Fields.website = normalised;
            try {
              const r = await runPageSpeedAudit(normalised);
              auditResult = {
                mobileScore:  r.mobileScore,
                desktopScore: r.desktopScore,
                summary:      r.summary,
                issues:       r.issues.slice(0, 2)
              };
              return auditResult;
            } catch {
              return { error: "Audit unavailable — proceeding." };
            }
          }
        })
      }
    });

    // If the model only called tools and produced no text (shouldn't happen with
    // maxSteps:2, but guard anyway), do a plain follow-up call with no tools.
    let responseText = result.text?.trim();
    if (!responseText) {
      const fallback = await generateText({
        model: getChatModel(true),
        system: BOT_SYSTEM_PROMPT,
        messages: [
          ...messages,
          // Summarise what was just stored so the AI knows where to continue
          {
            role: "assistant" as const,
            content: `[System: just stored fields: ${Object.keys(cf7Fields).join(", ") || "none"}]`
          }
        ],
        stopWhen: stepCountIs(1),
        providerOptions,
        maxOutputTokens: 120
      });
      responseText = fallback.text?.trim() || "Thanks! What's next on the form?";
    }

    const allLeadData = { ...leadData, ...cf7Fields };

    return NextResponse.json({
      text:      responseText,
      cf7Fields: Object.keys(cf7Fields).length > 0 ? cf7Fields : null,
      audit:     auditResult,
      leadData:  allLeadData
    });

  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    console.error("[BotChat]", detail);
    return NextResponse.json({ error: "Agent error", detail }, { status: 500 });
  }
}
