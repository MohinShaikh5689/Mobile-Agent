import { parseTimeWindow } from './timeWindow';
import type { Source, TimeWindow } from './types';

/**
 * The router classifies a request with keyword and date-pattern rules, not the model's own judgment.
 *
 * - structured: an exact lookup the API answers with filters (tomorrow's calendar, unread mail from Sam)
 * - fuzzy:      recall by meaning, answered from the hybrid index + live retrieval
 * - multi_step: spans services or chains steps; goes to the planner
 * - action:     asks for a write (draft, send, invite); needs the approval gate
 * - general:    small talk or a general question; answered by the model, no search
 *
 * When the rules cannot tell a general question from one about the user's data, the decision
 * says `ambiguous` and the graph asks the model a yes/no question (the "tiny classifier").
 */
export type Route = 'structured' | 'fuzzy' | 'multi_step' | 'action' | 'general';

export interface RouteDecision {
  route: Route;
  services: Source[];
  window: TimeWindow | null;
  /** Phrase that set the window, stripped from search keywords. */
  windowPhrase: string | null;
  /** Search keywords for live API queries. */
  keywords: string[];
  /** Why this route was chosen, for tracing. */
  reason: string;
  /** Rules could not tell general from personal; the graph's classify step decides. */
  ambiguous?: boolean;
}

const SERVICE_PATTERNS: Record<Source, RegExp> = {
  mail: /\b(e-?mails?|mails?|inbox|messages?|sent|send|wrote|replied|reply|replies|threads?|unread|cc'?d|draft|note to|tell|notify)\b/,
  docs: /\b(docs?|documents?|files?|drive|contracts?|sheets?|spreadsheets?|slides?|pdfs?|notes|proposals?|reports?)\b/,
  calendar: /\b(calendar|meetings?|events?|schedule[d]?|agenda|appointments?|busy|free|invites?|call with|1:1)\b/,
};

/** Imperative write requests. Verbs that also appear in questions ("what did Sam send") need an object after them. */
const OBJ = '(?:a|an|the|him|her|them|it|this|that|my|our|out|back)';
const ACTION_PATTERN = new RegExp(
  '\\b(?:draft|compose|forward|reschedule|cancel|decline|accept)\\b' +
    `|\\b(?:write|send|book|create|schedule|move|invite)\\s+${OBJ}\\b` +
    '|\\b(?:reply|respond)\\s+(?:to|back)\\b' +
    '|\\b(?:set up|add)\\s+(?:a|an)\\s+(?:event|meeting|call|reminder)\\b',
);

const QUESTION_START = /^\s*(what|what's|whats|did|has|have|had|when|who|where|which|is|are|was|were|do|does|how)\b/;
const POLITE_REQUEST = /\b(can you|could you|would you|please|help me)\b/;

/** Exact-filter phrasing: these are filters, not similarity. */
const STRUCTURED_PATTERN = new RegExp(
  "\\b(unread|starred|from\\s+[a-z]+|what(?:'s| is) on|agenda|next meeting|first meeting|how many|list|any meetings|am i free|do i have)\\b" +
    // "Did Sam reply?", "Has the bank emailed me?": a yes/no check on one sender.
    '|^\\s*(?:did|has|have)\\s+\\S+(?:\\s+\\S+)?\\s+(?:reply|replied|respond|responded|write|written|wrote|send|sent|e-?mail(?:ed)?)\\b',
);

/** Recall-by-meaning phrasing. */
const FUZZY_PATTERN =
  /\b(about|regarding|mention(?:ed|s)?|the (?:thread|email|doc|document|file|one) (?:about|with|where|that)|find the|what did .+ say|said about|related to|something about)\b/;

/** Greetings, thanks and questions about the assistant itself. */
const SMALLTALK =
  /^\s*(hi|hello|hey|hiya|yo|sup|good (morning|afternoon|evening|night)|thanks|thank you|thx|ty|ok|okay|cool|nice|great|awesome|bye|goodbye|see you|how are you|how's it going|how is it going|what's up|whats up|who are you|what are you|what can you do|help)\b/;

/** Words that tie a request to the user's own data. */
const PERSONAL =
  /\b(my|me|mine|i|i'm|i've|i'd|i'll|we|our|us|unread|starred|inbox|received|got any)\b|\b(did|has|have)\s+\S+\s+(say|said|send|sent|write|wrote|reply|replied|email|emailed|share|shared)\b/;

const CHAIN_PATTERN = /\b(and then|then|after that|and (?:also )?(?:draft|send|email|tell|notify|invite))\b/;

const STOPWORDS = new Set(
  (
    'a an the and or but of to in on at for from by with about regarding what whats which who whom when where why how ' +
    'is are was were be been being do does did have has had i me my mine you your we our us it its this that these those ' +
    'there their they them he she his her can could would should will shall may might must find show tell give get list ' +
    'any some all please email emails mail mails message messages thread doc docs document documents file files ' +
    'calendar meeting meetings event events say said sent send me up'
  ).split(' '),
);

export function extractKeywords(text: string, windowPhrase: string | null): string[] {
  let q = text.toLowerCase();
  if (windowPhrase) q = q.replace(windowPhrase, ' ');
  const words = q.match(/[a-z0-9][a-z0-9'@._-]*/g) ?? [];
  const out: string[] = [];
  for (const w of words) {
    const clean = w.replace(/'s$/, '').replace(/[.'_-]+$/, '');
    if (clean.length < 2 || STOPWORDS.has(clean) || out.includes(clean)) continue;
    out.push(clean);
  }
  return out;
}

export function detectServices(text: string): Source[] {
  const q = text.toLowerCase();
  return (Object.keys(SERVICE_PATTERNS) as Source[]).filter((s) => SERVICE_PATTERNS[s].test(q));
}

export function routeRequest(text: string, now: number): RouteDecision {
  const q = text.toLowerCase();
  const parsed = parseTimeWindow(q, now);
  const window = parsed?.window ?? null;
  const windowPhrase = parsed?.phrase ?? null;
  const keywords = extractKeywords(q, windowPhrase);
  let services = detectServices(q);

  // "Did Sam reply to my email?" is a question, not a request to reply.
  const isQuestion = QUESTION_START.test(q) && !POLITE_REQUEST.test(q);
  const isAction = ACTION_PATTERN.test(q) && !isQuestion;
  const chained = CHAIN_PATTERN.test(q);

  const base = { window, windowPhrase, keywords };

  if (isAction && (chained || services.length > 1)) {
    return { route: 'multi_step', services, ...base, reason: 'write action combined with another step or service' };
  }
  if (isAction) {
    return { route: 'action', services, ...base, reason: 'asks for a write action' };
  }
  if (services.length > 1 && chained) {
    return { route: 'multi_step', services, ...base, reason: 'chains steps across services' };
  }

  // Calendar questions are always exact range lookups; the index does not hold calendar data.
  if (services.length === 1 && services[0] === 'calendar') {
    return { route: 'structured', services, ...base, reason: 'calendar lookup is a range filter' };
  }

  const fuzzy = FUZZY_PATTERN.test(q);
  const structured = STRUCTURED_PATTERN.test(q);

  if (services.length === 0) {
    const words = q.split(/\s+/).filter(Boolean).length;
    if (SMALLTALK.test(q) && words <= 8) {
      return { route: 'general', services: [], ...base, reason: 'small talk' };
    }
    // Without a service word or a personal cue ("my", "did Sam send"), the patterns below are too
    // broad ("list some books", "where does pasta come from"): let the classifier decide.
    if (!PERSONAL.test(q)) {
      return {
        route: 'fuzzy',
        services: ['mail', 'docs'],
        ...base,
        ambiguous: true,
        reason: 'no service or personal cue; classifier decides between general and recall',
      };
    }
  }

  if (structured && !fuzzy) {
    if (services.length === 0) services = ['mail'];
    return { route: 'structured', services, ...base, reason: 'exact filter phrasing' };
  }
  if (!fuzzy && window && keywords.length <= 1) {
    if (services.length === 0) services = ['mail'];
    return { route: 'structured', services, ...base, reason: 'date filter with no topic' };
  }

  // Fuzzy recall: search mail and docs unless the request names one.
  const recallServices = services.filter((s) => s !== 'calendar');
  return {
    route: 'fuzzy',
    services: recallServices.length > 0 ? recallServices : ['mail', 'docs'],
    ...base,
    reason: fuzzy ? 'recall-by-meaning phrasing' : 'no exact filter found; default to recall',
  };
}
