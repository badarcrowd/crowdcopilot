import type { LeadFormData } from "@/lib/agent/lead-score";

export type LeadValidationResult = {
  errors: Record<string, string>;
  normalized: Partial<Record<keyof LeadFormData, string>>;
};

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;
const PHONE_REGEX = /^\+[1-9]\d{6,14}$/;
const LEAD_FIELD_KEYS = new Set<string>([
  "first_name",
  "last_name",
  "company",
  "sector",
  "challenge",
  "success",
  "budget",
  "location",
  "start",
  "rfp",
]);

export const CORE_INTAKE_FIELDS: Array<keyof LeadFormData> = [
  "website",
  "email",
  "phone",
];

export function splitFullName(name: string): Pick<LeadFormData, "first_name" | "last_name" | "name"> {
  const clean = name.trim().replace(/\s+/g, " ");
  const parts = clean.split(" ").filter(Boolean);
  return {
    name: clean,
    first_name: parts[0],
    last_name: parts.slice(1).join(" ") || undefined,
  };
}

export function normalizeWebsite(value: string) {
  return /^https?:\/\//i.test(value.trim()) ? value.trim() : `https://${value.trim()}`;
}

function normalizePhone(value: string) {
  return value
    .trim()
    .replace(/[()\-.]/g, " ")
    .replace(/\s+/g, " ")
    .replace(/^00/, "+");
}

function e164Candidate(value: string) {
  const phone = normalizePhone(value);
  if (!phone.startsWith("+")) return "";
  return `+${phone.slice(1).replace(/\D/g, "")}`;
}

function normalizeEmail(value: string) {
  return value
    .trim()
    .replace(/^mailto:/i, "")
    .replace(/\s*\(at\)\s*/gi, "@")
    .replace(/\s*\[at\]\s*/gi, "@")
    .replace(/\s+at\s+/gi, "@")
    .replace(/\s*\(dot\)\s*/gi, ".")
    .replace(/\s*\[dot\]\s*/gi, ".")
    .replace(/\s+dot\s+/gi, ".")
    .replace(/\s+/g, "")
    .replace(/[),.;]+$/g, "")
    .toLowerCase();
}

function normalizeSpokenWebsite(value: string) {
  return value
    .trim()
    .replace(/\s+dot\s+/gi, ".")
    .replace(/\s*\(dot\)\s*/gi, ".")
    .replace(/\s*\[dot\]\s*/gi, ".")
    .replace(/\s+slash\s+/gi, "/")
    .replace(/\s+/g, "")
    .replace(/[),.;]+$/g, "");
}

function validateWebsite(value: string) {
  const normalized = normalizeWebsite(normalizeSpokenWebsite(value));
  const url = new URL(normalized);
  if (!url.hostname.includes(".") || url.hostname.length > 253) {
    throw new Error("Invalid hostname");
  }
  return url.toString();
}

export function validateLeadFields(fields: Record<string, unknown>): LeadValidationResult {
  const errors: Record<string, string> = {};
  const normalized: Partial<Record<keyof LeadFormData, string>> = {};

  for (const [key, rawValue] of Object.entries(fields)) {
    if (typeof rawValue !== "string") continue;
    const value = rawValue.trim();
    if (!value) continue;

    if (key === "name") {
      Object.assign(normalized, splitFullName(value));
      continue;
    }

    if (key === "email") {
      const email = normalizeEmail(value);
      if (!EMAIL_REGEX.test(email)) {
        errors.email = "That email looks incomplete — could you resend just your work email?";
        continue;
      }
      normalized.email = email;
      continue;
    }

    if (key === "phone") {
      const phone = normalizePhone(value);
      const e164 = e164Candidate(phone);
      if (!phone.startsWith("+") || !PHONE_REGEX.test(e164)) {
        errors.phone = "That number looks incomplete — could you resend it with country code?";
        continue;
      }
      normalized.phone = phone;
      continue;
    }

    if (key === "website") {
      try {
        normalized.website = validateWebsite(value);
      } catch {
        errors.website = "That website does not look quite right — could you resend just the URL?";
      }
      continue;
    }

    if (LEAD_FIELD_KEYS.has(key)) {
      normalized[key as keyof LeadFormData] = value;
    }
  }

  return { errors, normalized };
}

export function createCf7Fields(fields: Partial<Record<keyof LeadFormData, string>>) {
  const cf7Fields: Record<string, string> = {};
  if (fields.first_name) cf7Fields.first_name = fields.first_name;
  if (fields.last_name) cf7Fields.last_name = fields.last_name;
  if (fields.name && !cf7Fields.first_name) {
    const split = splitFullName(fields.name);
    if (split.first_name) cf7Fields.first_name = split.first_name;
    if (split.last_name) cf7Fields.last_name = split.last_name;
  }
  if (fields.email) cf7Fields.email = fields.email;
  if (fields.phone) cf7Fields.phone = fields.phone;
  if (fields.company) cf7Fields.company = fields.company;
  if (fields.website) cf7Fields.website = fields.website;
  if (fields.sector) cf7Fields.sector = fields.sector;
  if (fields.challenge) cf7Fields.business = fields.challenge;
  if (fields.success) cf7Fields.success = fields.success;
  if (fields.budget) cf7Fields.cost = fields.budget;
  if (fields.start) cf7Fields.start = fields.start;
  if (fields.location) cf7Fields.location = fields.location;
  if (fields.rfp) cf7Fields.rfp = fields.rfp;
  return cf7Fields;
}

export function missingCoreFields(leadData: Partial<LeadFormData>) {
  return CORE_INTAKE_FIELDS.filter((field) => {
    if (field === "first_name" || field === "last_name") {
      return !leadData.name && !leadData.first_name;
    }
    return !leadData[field];
  });
}

export function extractLeadFieldsFromText(text: string): Partial<LeadFormData> {
  const extracted: Partial<LeadFormData> = {};
  const source = text.trim();

  const email = source.match(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i)?.[0];
  if (email) extracted.email = email;

  const sourceWithoutEmail = email ? source.replace(email, "") : source;
  const website = sourceWithoutEmail.match(/\b((?:https?:\/\/)?(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:\/[^\s,;]*)?)/i)?.[0];
  if (website && !website.includes("@")) extracted.website = website;

  const phone = source.match(/(?:\+\d{1,3}|00\d{1,3})[\d\s().-]{6,}\d/)?.[0];
  if (phone) extracted.phone = phone;

  const withoutKnown = source
    .replace(email ?? "", "")
    .replace(website ?? "", "")
    .replace(phone ?? "", "")
    .replace(/\b(my name is|i am|i'm|this is|name is)\b/gi, "")
    .split(/[,;\n]/)
    .map((part) => part.trim())
    .filter(Boolean);

  const nameCandidate = withoutKnown.find((part) =>
    /^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}$/.test(part)
  );
  if (nameCandidate) Object.assign(extracted, splitFullName(nameCandidate));

  return extracted;
}
