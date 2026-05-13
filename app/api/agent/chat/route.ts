import { streamText, tool, stepCountIs, convertToCoreMessages } from "ai";
import { NextResponse } from "next/server";
import { z } from "zod";
import type { OpenAIResponsesProviderOptions } from "@ai-sdk/openai";
import { getChatModel, getEmbeddingModel } from "@/lib/ai/openai";
import { runPageSpeedAudit } from "@/lib/agent/pagespeed";
import { scoreLead, evaluateBudget } from "@/lib/agent/lead-score";
import { getAvailableSlots, formatSlotsForAgent } from "@/lib/agent/calendar";
import { syncLeadToMonday } from "@/lib/agent/monday";
import { rateLimit } from "@/lib/security/rate-limit";
import { createSupabaseAdmin } from "@/lib/supabase";
import { agentUseSmallModel } from "@/lib/env";
import { sanitizeTenantId } from "@/lib/rag";
import { embed } from "ai";
import type { LeadFormData } from "@/lib/agent/lead-score";

export const runtime = "nodejs";
export const maxDuration = 60;

const agentRequestSchema = z.object({
  messages: z.array(z.any()).min(1),
  sessionId: z.string().optional(),
  tenantId: z.string().optional().nullable(),
  leadData: z
    .object({
      name: z.string().optional(),
      email: z.string().optional(),
      company: z.string().optional(),
      website: z.string().optional(),
      sector: z.string().optional(),
      challenge: z.string().optional(),
      budget: z.string().optional(),
      phone: z.string().optional(),
      first_name: z.string().optional(),
      last_name: z.string().optional(),
      location: z.string().optional(),
      success: z.string().optional(),
      start: z.string().optional(),
      rfp: z.string().optional()
    })
    .optional()
});

const providerOptions: { openai: OpenAIResponsesProviderOptions } = {
  openai: {
    reasoningEffort: "low",
    textVerbosity: "low",
    promptCacheKey: "crowd-agent-onboarding",
    promptCacheRetention: "in_memory"
  }
};

const AGENT_SYSTEM_PROMPT = `You are Crowd Agent — a magical, animated, voice-enabled AI onboarding assistant for Crowd Digital, a premium digital growth agency.

CRITICAL — TTS OUTPUT RULES (your responses are spoken aloud):
- NO markdown, no bullet dashes, no asterisks, no hashtags, no backticks
- Keep every response under 3 short sentences maximum
- Write in natural spoken English only
- Never list multiple questions in one turn — one thing at a time always

PERSONA:
- Warm, elegant, slightly playful, and deeply professional
- You feel magical and futuristic — like talking to a brilliant consultant who also happens to be brilliant company
- Use micro-acknowledgments naturally: "Got it", "Perfect", "Nice", "Love that"
- Never sound robotic or form-like
- If user hesitates, gently guide them forward

CONVERSATIONAL FLOW — sound like a strategist, never a survey or form:

STEP 1 — OPENER (first message of the conversation only):
Lead with one warm, consultative line that asks for the three essentials together — website, work email, contact number with country code — framed around the value the visitor gets ("so I can run a quick analysis"). Never use "please provide…", comma-separated field lists, or rigid form phrasing.
Good opener: "Happy to help. I can run a quick analysis for you — just send over your website, work email, and contact number with country code, and I'll take it from there."
Do NOT ask for full name upfront. Do NOT list every field. Do NOT sound like a checklist.

STEP 2 — WHEN WEBSITE IS CAPTURED:
Silently call run_website_audit immediately. Continue the conversation naturally — do not announce technical processing. Once the audit returns, drop ONE useful finding in conversational language.
If the visitor gave several fields at once, call extract_lead_data ONCE with every field you can identify and continue from the earliest missing field. Never re-ask captured fields.

STEP 3 — PROGRESSIVE DISCLOSURE (one casual question per reply):
"And which company are you working with?" → company
"By the way, what industry are you in?" → sector
"Which region should we route this through — Middle East, Europe, Asia, or the US?" → location
First name only when it feels natural or before handoff.

STEP 4 — BUSINESS NEEDS (two separate turns):
First ask: "What's the core business or marketing challenge you're looking to solve?"
After answer → call extract_lead_data with challenge, acknowledge briefly, then ask:
"What does success look like for this project?"

STEP 5 — PROJECT DETAILS (three separate turns):
Ask: "What's your available budget for this project?"
When given → call validate_budget, call extract_lead_data with budget.
Then ask: "When are you looking to start?"
Then ask: "Is this part of an RFP process?"
If yes → ask: "How many agencies are you considering?" then "Are you working with an incumbent agency?"
Call extract_lead_data with rfp value.

STEP 6 — BOOKING:
Once all key fields collected OR lead score 70+ — call get_calendar_slots.
Say: "Perfect. I've got everything I need. Let's find a time that works for you."
Present available slots conversationally, not as a list.
After booking confirmed: "You're all set. Looking forward to connecting with you."

TOOL RULES — call proactively, never wait:
- Any name, email, phone, company, website, sector, challenge, budget, location, rfp mentioned → call extract_lead_data immediately
- Website URL given → call run_website_audit immediately  
- Sector mentioned → call search_case_studies
- Challenge or success goal mentioned → call analyze_challenge
- Budget stated → call validate_budget
- Challenge + sector, budget, or audit context available → call generate_business_insight
- Have name + email + one more field → call score_lead
- Score 70+ OR booking intent → call get_calendar_slots
- Have name + email + budget or audit → call sync_to_crm

SOFT VALIDATION — never say "invalid":
- If a tool returns success:false with validationErrors, ask gently for the corrected field only — never move on.
- Phone missing country code → "That number looks incomplete — could you resend it with country code?"
- Email unclear → "Hmm, that email doesn't look right — mind double-checking it?"
- URL unclear → "That URL doesn't look right — try something like https://yoursite.com"
- If the user gives multiple fields at once, acknowledge them and continue from the earliest missing field.
- Never expose lead scores as raw scoring math unless asked. Use the score to decide urgency and booking priority.

BUDGET GUIDANCE (spoken naturally):
Under 5k: gently suggest our free audit starter report.
5k to 15k: focused sprint on SEO or paid media.
15k to 50k: multi-channel growth programme.
50k plus: full enterprise programme with a dedicated team.

Never dump multiple questions at once. Always feel fluid, human, and slightly magical.`;

function normalizeWebsite(value: string) {
  return /^https?:\/\//i.test(value) ? value : `https://${value}`;
}

function validateLeadFields(fields: Record<string, unknown>) {
  const errors: Record<string, string> = {};
  const normalized: Record<string, string> = {};

  for (const [key, rawValue] of Object.entries(fields)) {
    if (typeof rawValue !== "string") continue;
    const value = rawValue.trim();
    if (!value) continue;

    if (key === "email") {
      if (!/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(value)) {
        errors.email = "That email does not look right. Could you double-check it?";
        continue;
      }
    }

    if (key === "phone") {
      const digits = value.replace(/\D/g, "");
      if (digits.length < 7) {
        errors.phone = "That phone number looks too short. Could you include the country code?";
        continue;
      }
    }

    if (key === "website") {
      try {
        normalized.website = new URL(normalizeWebsite(value)).toString();
        continue;
      } catch {
        errors.website = "That website URL does not look right. Try something like https://example.com.";
        continue;
      }
    }

    normalized[key] = value;
  }

  return { errors, normalized };
}

function createCf7Fields(fields: Record<string, string>) {
  const cf7Fields: Record<string, string> = {};
  if (fields.first_name) cf7Fields.first_name = fields.first_name;
  if (fields.last_name)  cf7Fields.last_name  = fields.last_name;
  if (fields.email)      cf7Fields.email      = fields.email;
  if (fields.phone)      cf7Fields.phone      = fields.phone;
  if (fields.company)    cf7Fields.company    = fields.company;
  if (fields.website)    cf7Fields.website    = fields.website;
  if (fields.sector)     cf7Fields.sector     = fields.sector;
  if (fields.challenge)  cf7Fields.business   = fields.challenge;
  if (fields.success)    cf7Fields.success    = fields.success;
  if (fields.budget)     cf7Fields.cost       = fields.budget;
  if (fields.start)      cf7Fields.start      = fields.start;
  if (fields.location)   cf7Fields.location   = fields.location;
  if (fields.rfp)        cf7Fields.rfp        = fields.rfp;
  return cf7Fields;
}

function generateBusinessInsight(params: {
  sector?: string;
  challenge?: string;
  budget?: string;
  auditScore?: number;
}) {
  const score = scoreLead(
    {
      sector: params.sector,
      challenge: params.challenge,
      budget: params.budget
    },
    params.auditScore
  );
  const budget = params.budget ? evaluateBudget(params.budget) : null;
  const actions = [
    params.auditScore !== undefined && params.auditScore < 65
      ? "Prioritize landing-page speed and conversion fixes before scaling media spend."
      : "Use the current website as the conversion hub and validate the best channel mix.",
    params.sector
      ? `Anchor recommendations in ${params.sector} benchmarks and relevant case studies.`
      : "Confirm sector fit so recommendations can be benchmarked properly.",
    score.tier === "hot"
      ? "Offer a strategy call now while intent is high."
      : "Share one useful audit or case-study insight before asking for a meeting."
  ];

  return {
    tier: score.tier,
    score: score.score,
    bookingPriority: score.bookingPriority,
    budgetGuidance: budget?.suggestion ?? "Ask for a rough budget range before recommending a programme.",
    actions
  };
}

async function searchCaseStudies(
  sector: string,
  tenantId: string
): Promise<string> {
  try {
    const supabase = createSupabaseAdmin();
    const { embedding } = await embed({
      model: getEmbeddingModel(),
      value: `${sector} case study results growth marketing`
    });

    const { data } = await supabase.rpc("match_chunks", {
      query_embedding: embedding,
      match_count: 3,
      tenant_filter: tenantId,
      min_similarity: 0.2
    });

    if (!data?.length) {
      return `No specific case studies found for ${sector}, but we have extensive experience across digital sectors.`;
    }

    return data
      .map((c: { content: string }) => c.content)
      .join("\n\n---\n\n")
      .slice(0, 1200);
  } catch {
    return `We have delivered 40–300% growth for ${sector} clients through performance marketing and CRO.`;
  }
}

async function recordAgentSession(
  sessionId: string,
  tenantId: string,
  leadData: LeadFormData
) {
  try {
    const supabase = createSupabaseAdmin();
    await supabase.from("lead_agent_sessions").upsert(
      {
        id: sessionId,
        tenant_id: tenantId,
        form_data: leadData,
        updated_at: new Date().toISOString()
      },
      { onConflict: "id" }
    );
  } catch {
    // Non-critical — don't block the response
  }
}

export async function POST(request: Request) {
  const limited = await rateLimit(request, "agent", 20, 60);
  if (limited) return limited;

  const json = await request.json();
  const parsed = agentRequestSchema.safeParse(json);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Invalid request", details: parsed.error.flatten() },
      { status: 400 }
    );
  }

  const { messages: rawMessages, leadData = {} } = parsed.data;
  const sessionId = parsed.data.sessionId ?? crypto.randomUUID();
  const tenantId = sanitizeTenantId(parsed.data.tenantId ?? null);

  // Convert UIMessages (parts format from useChat) to CoreMessages for streamText
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const messages = convertToCoreMessages(rawMessages as any[]);

  // Persist session asynchronously
  if (Object.keys(leadData).length > 0) {
    void recordAgentSession(sessionId, tenantId, leadData as LeadFormData);
  }

  const result = streamText({
    model: getChatModel(agentUseSmallModel()),
    system: AGENT_SYSTEM_PROMPT,
    messages,
    stopWhen: stepCountIs(5), // Allow up to 5 sequential LLM → tool → LLM steps
    providerOptions,
    maxOutputTokens: 220,
    tools: {
      run_website_audit: tool({
        description:
          "Run a PageSpeed + SEO audit on a website URL. Call this whenever the user provides a URL.",
        inputSchema: z.object({
          url: z.string().describe("The website URL to audit")
        }),
        execute: async ({ url }) => {
          try {
            const result = await runPageSpeedAudit(normalizeWebsite(url));
            return {
              success: true,
              url: result.url,
              mobileScore: result.mobileScore,
              desktopScore: result.desktopScore,
              performanceScore: result.performanceScore,
              issues: result.issues,
              metrics: result.metrics,
              summary: result.summary
            };
          } catch (err) {
            return {
              success: false,
              error: err instanceof Error ? err.message : "Audit failed"
            };
          }
        }
      }),

      search_case_studies: tool({
        description:
          "Search for relevant case studies based on the prospect's sector/industry. Use this when the user mentions their industry.",
        inputSchema: z.object({
          sector: z
            .string()
            .describe("The prospect's industry or sector (e.g. ecommerce, SaaS)")
        }),
        execute: async ({ sector }) => {
          const content = await searchCaseStudies(sector, tenantId);
          return { sector, content };
        }
      }),

      analyze_challenge: tool({
        description:
          "Analyse the prospect's business challenge. Call this when the user describes their problem, success goal, timeline, or growth target.",
        inputSchema: z.object({
          challenge: z
            .string()
            .describe("The prospect's main business challenge or goal")
        }),
        execute: async ({ challenge }) => {
          // Let the LLM generate the analysis in its response text
          return {
            challenge,
            structuredProblem: challenge.slice(0, 200),
            timestamp: new Date().toISOString()
          };
        }
      }),

      generate_business_insight: tool({
        description:
          "Create budget guidance and next-best actions from the visitor's sector, challenge, budget, and optional website audit score. Call once the user has shared a challenge plus sector, budget, or audit context.",
        inputSchema: z.object({
          sector: z.string().optional(),
          challenge: z.string().optional(),
          budget: z.string().optional(),
          auditScore: z.number().optional()
        }),
        execute: async (params) => generateBusinessInsight(params)
      }),

      validate_budget: tool({
        description:
          "Validate the prospect's budget against service tiers and suggest the best plan. Call this when any budget figure is mentioned.",
        inputSchema: z.object({
          budget: z.string().describe("The prospect's stated budget (e.g. '15k', '$20,000/month')")
        }),
        execute: async ({ budget }) => {
          const evaluation = evaluateBudget(budget);
          return evaluation;
        }
      }),

      score_lead: tool({
        description:
          "Calculate a lead score based on all collected data. Call this when you have name + email + at least one other field.",
        inputSchema: z.object({
          name: z.string().optional(),
          email: z.string().optional(),
          company: z.string().optional(),
          website: z.string().optional(),
          sector: z.string().optional(),
          challenge: z.string().optional(),
          success: z.string().optional(),
          budget: z.string().optional(),
          start: z.string().optional(),
          rfp: z.string().optional(),
          auditScore: z
            .number()
            .optional()
            .describe("The website performance score from audit, if available")
        }),
        execute: async (params) => {
          const { auditScore, ...formFields } = params;
          const result = scoreLead(formFields, auditScore);
          return result;
        }
      }),

      get_calendar_slots: tool({
        description:
          "Fetch available meeting slots. Call this when the user expresses booking intent OR when the lead score is 70+.",
        inputSchema: z.object({
          preferredTime: z
            .enum(["morning", "afternoon", "any"])
            .optional()
            .default("any")
        }),
        execute: async () => {
          const slots = await getAvailableSlots(9);
          const formatted = formatSlotsForAgent(slots);
          return {
            slots: slots.slice(0, 6).map((s) => ({
              id: s.id,
              displayDate: s.displayDate,
              displayTime: s.displayTime,
              isoDateTime: s.isoDateTime
            })),
            formattedList: formatted
          };
        }
      }),

      extract_lead_data: tool({
        description:
          "Extract and store lead information from the conversation. Call this IMMEDIATELY whenever the user mentions their name, email, phone, company, website, industry, challenge, budget, office/location, or RFP details — even partial information.",
        inputSchema: z.object({
          first_name: z.string().optional().describe("Prospect's first name"),
          last_name: z.string().optional().describe("Prospect's last name"),
          name: z.string().optional().describe("Prospect's full name (will be split into first/last)"),
          email: z.string().optional().describe("Prospect's email address"),
          phone: z.string().optional().describe("Prospect's phone number"),
          company: z.string().optional().describe("Company or business name"),
          website: z.string().optional().describe("Website URL"),
          sector: z.string().optional().describe("Industry or sector"),
          challenge: z.string().optional().describe("Main business challenge or goal"),
          success: z.string().optional().describe("What success looks like for the project"),
          budget: z.string().optional().describe("Marketing budget stated by the user"),
          start: z.string().optional().describe("Desired start date or launch timing"),
          location: z.string().optional().describe("Preferred Crowd office / region"),
          rfp: z.string().optional().describe("RFP details — yes/no and number of agencies, incumbent agency")
        }),
        execute: async (params) => {
          // Split full name into first/last if provided
          if (params.name && !params.first_name) {
            const parts = params.name.trim().split(/\s+/);
            params.first_name = parts[0];
            params.last_name = parts.slice(1).join(" ") || undefined;
          }

          const { errors, normalized } = validateLeadFields(params);
          const cf7Fields = createCf7Fields(normalized);

          // Persist updated lead data to session
          if (Object.keys(normalized).length > 0) {
            const updated = { ...leadData, ...normalized };
            void recordAgentSession(sessionId, tenantId, updated as LeadFormData);
          }

          return {
            extracted: normalized,
            cf7Fields,
            success: Object.keys(errors).length === 0,
            validationErrors: Object.keys(errors).length > 0 ? errors : undefined
          };
        }
      }),

      sync_to_crm: tool({
        description:
          "Sync the qualified lead to the CRM (Monday.com). Call this once name + email are collected AND either budget or audit is complete.",
        inputSchema: z.object({
          name: z.string(),
          email: z.string().email(),
          phone: z.string().optional(),
          company: z.string().optional(),
          website: z.string().optional(),
          sector: z.string().optional(),
          budget: z.string().optional(),
          challenge: z.string().optional(),
          leadScore: z.number().min(0).max(100),
          leadTier: z.enum(["hot", "warm", "cold"]),
          aiSummary: z
            .string()
            .describe("1–2 sentence AI-generated summary of this lead")
        }),
        execute: async (params) => {
          const { leadScore, leadTier, aiSummary, ...formFields } = params;
          const scoreResult = scoreLead(formFields, undefined);

          try {
            const supabase = createSupabaseAdmin();
            await supabase.from("leads").upsert(
              {
                tenant_id: tenantId,
                session_id: sessionId,
                name: params.name,
                email: params.email,
                phone: params.phone ?? null,
                status: leadTier === "hot" ? "hot_lead" : "new",
                metadata: {
                  company: params.company,
                  website: params.website,
                  sector: params.sector,
                  budget: params.budget,
                  challenge: params.challenge,
                  lead_score: leadScore,
                  lead_tier: leadTier,
                  ai_summary: aiSummary
                }
              },
              { onConflict: "tenant_id, email" }
            );

            const monday = await syncLeadToMonday({
              form: formFields,
              score: scoreResult,
              aiSummary,
              sessionId
            });

            return {
              success: true,
              leadScore,
              leadTier,
              mondayId: monday.id,
              message: "Lead saved and synced to CRM."
            };
          } catch (err) {
            return {
              success: false,
              error: err instanceof Error ? err.message : "CRM sync failed"
            };
          }
        }
      })
    },

    onError: (error) => {
      console.error("[Agent Error]", error);
    }
  });

  return result.toUIMessageStreamResponse();
}
