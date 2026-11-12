import type { ProposedAction } from './actions';
import type { TraceStep } from './agent';
import type { ChatModel } from './llm/types';
import type { KeyValueStore } from './services';
import type { ScoredItem, Source } from './types';

/**
 * Conversation memory sized for a small context window.
 *
 * Each turn the model sees, within a fixed token budget:
 * - a running summary of older turns,
 * - the items (emails, docs, events) mentioned so far, by id, so "that email" can be re-fetched,
 * - the most recent turns verbatim.
 * When the verbatim turns outgrow their share, the oldest are folded into the summary.
 */

export interface Turn {
  role: 'user' | 'assistant';
  text: string;
  at: number;
}

export interface RefItem {
  source: Source;
  id: string;
  title: string;
  date: string;
}

export interface Conversation {
  id: string;
  createdAt: number;
  turns: Turn[];
  summary: string;
  refs: RefItem[];
  /** Services of the last data question, so follow-ups without a service word stay on topic. */
  lastServices?: Source[];
  /** A proposed write waiting for approval (the paused graph thread and what it will do). */
  pending?: { threadId: string; action: ProposedAction };
}

/** Rough token count for English text (about 3.5 characters per token). */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 3.5);
}

/**
 * Tokens of history per turn: about a fifth of the context, the rest stays for instructions,
 * tools, search results and the answer. 4K → 900, 8K → 1.8K, 16K → 3.6K, 32K → 6K (capped:
 * beyond that the summary carries older turns and prefill time on a phone matters more).
 */
export function historyBudget(nCtx: number): number {
  return Math.min(6000, Math.max(900, Math.floor(nCtx * 0.22)));
}

const MAX_REFS = 10;
const TURN_CHARS = 600;
/** Always keep at least this many recent messages verbatim. */
const KEEP_RECENT = 4;

function clip(text: string, max: number): string {
  const t = text.replace(/\s+/g, ' ').trim();
  return t.length > max ? `${t.slice(0, max - 1)}…` : t;
}

/** The conversation as text for the prompt, within `budget` tokens. Empty for a new conversation. */
export function renderHistory(conv: Conversation | null, budget: number): string {
  if (!conv || (conv.turns.length === 0 && !conv.summary)) return '';
  const parts: string[] = [];
  if (conv.summary) parts.push(`Earlier in this conversation: ${conv.summary}`);
  if (conv.refs.length) {
    parts.push(
      'Items mentioned so far (newest first; use the id with a read tool):\n' +
        conv.refs.map((r) => `- ${r.source} id=${r.id} | ${r.date} | ${clip(r.title, 80)}`).join('\n'),
    );
  }
  let used = estimateTokens(parts.join('\n'));
  const recent: string[] = [];
  for (let i = conv.turns.length - 1; i >= 0; i--) {
    const t = conv.turns[i] as Turn;
    const line = `${t.role === 'user' ? 'User' : 'Assistant'}: ${clip(t.text, TURN_CHARS)}`;
    const cost = estimateTokens(line);
    if (used + cost > budget && recent.length > 0) break;
    recent.unshift(line);
    used += cost;
  }
  if (recent.length) parts.push(`Recent messages:\n${recent.join('\n')}`);
  return parts.join('\n\n');
}

const TOOL_SOURCE: Record<string, Source> = {
  search_mail: 'mail',
  read_message: 'mail',
  search_docs: 'docs',
  read_doc: 'docs',
  list_events: 'calendar',
  read_event: 'calendar',
};

/** Items the answer was built on: retrieval sources plus anything a tool listed or read. */
export function extractRefs(sources: ScoredItem[], steps: TraceStep[]): RefItem[] {
  const out: RefItem[] = [];
  const add = (r: RefItem) => {
    if (!out.some((x) => x.source === r.source && x.id === r.id)) out.push(r);
  };
  for (const s of sources) {
    add({ source: s.source, id: s.id, title: s.title, date: new Date(s.date).toISOString().slice(0, 10) });
  }
  for (const step of steps) {
    if (step.kind !== 'tool' || !step.ok) continue;
    const source = TOOL_SOURCE[step.name];
    if (!source) continue;
    // List lines: "id=X | 2026-11-10 09:30 | from: ... | Title"; full items: "id=X\nTitle: ...\n...Date: ..."
    for (const m of step.output.matchAll(/id=([\w.@:-]+) \| (\d{4}-\d{2}-\d{2})[^|\n]*\|(?: from: [^|\n]*\|)? ([^\n]+)/g)) {
      add({ source, id: m[1] as string, date: m[2] as string, title: (m[3] as string).trim() });
    }
    const full = step.output.match(/id=([\w.@:-]+)\nTitle: ([^\n]+)[\s\S]*?Date: (\d{4}-\d{2}-\d{2})/);
    if (full) add({ source, id: full[1] as string, title: (full[2] as string).trim(), date: full[3] as string });
  }
  return out;
}

/** Newest refs first, without duplicates, capped. */
export function mergeRefs(newer: RefItem[], older: RefItem[]): RefItem[] {
  const out: RefItem[] = [];
  for (const r of [...newer, ...older]) {
    if (!out.some((x) => x.source === r.source && x.id === r.id)) out.push(r);
  }
  return out.slice(0, MAX_REFS);
}

const FOLLOW_UP =
  /\b(it|its|that|this|those|these|they|them|their|he|she|him|her|his|hers|there|then|same|again|more|also|another|other|else|previous|last one|first one|second|third|above|earlier|before that|after that|what about|how about)\b/;

/** Whether a question leans on earlier turns and should be rewritten as a standalone question. */
export function needsContext(question: string): boolean {
  const q = question.toLowerCase().trim();
  if (q.split(/\s+/).length <= 4) return true;
  return FOLLOW_UP.test(q) || /^(and|but|so|ok|okay|also|what about|how about)\b/.test(q);
}

/**
 * Folds the oldest turns into the summary once the verbatim turns outgrow their share of the
 * budget. If the model call fails, the oldest turns are dropped instead: the conversation never
 * grows past the budget.
 */
export async function compact(conv: Conversation, model: ChatModel, budget: number): Promise<Conversation> {
  const turnsShare = Math.floor(budget * 0.6);
  const turnTokens = (ts: Turn[]) => ts.reduce((n, t) => n + estimateTokens(clip(t.text, TURN_CHARS)) + 3, 0);
  if (turnTokens(conv.turns) <= turnsShare || conv.turns.length <= KEEP_RECENT) return conv;

  let cut = 0;
  while (conv.turns.length - cut > KEEP_RECENT && turnTokens(conv.turns.slice(cut)) > turnsShare) cut += 2;
  const old = conv.turns.slice(0, cut);
  const keep = conv.turns.slice(cut);

  let summary = conv.summary;
  try {
    const res = await model.chat([
      {
        role: 'system',
        content:
          'Update the running summary of a conversation between a user and their email/calendar/docs assistant. ' +
          'Keep names, dates, decisions and what the user is looking for. At most 80 words. Reply with the summary only.',
      },
      {
        role: 'user',
        content:
          `Current summary: ${conv.summary || '(none)'}\n\nOlder messages to fold in:\n` +
          old.map((t) => `${t.role === 'user' ? 'User' : 'Assistant'}: ${clip(t.text, TURN_CHARS)}`).join('\n'),
      },
    ]);
    summary = clip(res.content, 600);
  } catch {
    // Keep the old summary; the folded turns are dropped.
  }
  return { ...conv, summary, turns: keep };
}

const convKey = (id: string) => `conversation:${id}`;
const CURRENT = 'conversation.current';

export class ConversationStore {
  constructor(private kv: KeyValueStore) {}

  async get(id: string): Promise<Conversation | null> {
    const raw = await this.kv.get(convKey(id));
    return raw ? (JSON.parse(raw) as Conversation) : null;
  }

  async save(conv: Conversation): Promise<void> {
    await this.kv.set(convKey(conv.id), JSON.stringify(conv));
  }

  /** The open conversation, created if there is none. */
  async current(now: number): Promise<Conversation> {
    const id = await this.kv.get(CURRENT);
    const existing = id ? await this.get(id) : null;
    return existing ?? this.start(now);
  }

  /** Starts a new conversation and makes it the open one. The previous one is deleted. */
  async start(now: number): Promise<Conversation> {
    const prev = await this.kv.get(CURRENT);
    if (prev) await this.kv.delete(convKey(prev));
    const conv: Conversation = {
      id: `conv-${now}-${Math.random().toString(36).slice(2, 8)}`,
      createdAt: now,
      turns: [],
      summary: '',
      refs: [],
    };
    await this.save(conv);
    await this.kv.set(CURRENT, conv.id);
    return conv;
  }
}
