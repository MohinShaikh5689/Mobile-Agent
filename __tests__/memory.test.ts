import type { TraceStep } from '../src/core/agent';
import { autoContext, contextOptions, effectiveContext, findModel } from '../src/core/llm/models';
import type { ChatMessage, ChatModel, ChatResult } from '../src/core/llm/types';
import type { Conversation } from '../src/core/memory';
import {
  compact,
  ConversationStore,
  estimateTokens,
  extractRefs,
  historyBudget,
  mergeRefs,
  needsContext,
  renderHistory,
} from '../src/core/memory';
import { Agent } from '../src/core/orchestrator';
import { HybridRetriever } from '../src/core/retriever';
import type { KeyValueStore, Services } from '../src/core/services';
import type { Item } from '../src/core/types';
import { InMemoryVectorIndex } from '../src/core/vectorIndex';

class MemoryKV implements KeyValueStore {
  data = new Map<string, string>();
  async get(k: string) {
    return this.data.get(k) ?? null;
  }
  async set(k: string, v: string) {
    this.data.set(k, v);
  }
  async delete(k: string) {
    this.data.delete(k);
  }
}

const conv = (n: number, extra: Partial<Conversation> = {}): Conversation => ({
  id: 'c',
  createdAt: 0,
  summary: '',
  refs: [],
  turns: Array.from({ length: n }, (_, i) => ({
    role: i % 2 === 0 ? ('user' as const) : ('assistant' as const),
    text: `message ${i} ${'word '.repeat(40)}`,
    at: i,
  })),
  ...extra,
});

describe('history rendering', () => {
  it('is empty for a new conversation', () => {
    expect(renderHistory(null, 900)).toBe('');
    expect(renderHistory(conv(0), 900)).toBe('');
  });

  it('keeps the newest turns within the budget, oldest dropped first', () => {
    const text = renderHistory(conv(40), 900);
    expect(estimateTokens(text)).toBeLessThanOrEqual(900);
    expect(text).toContain('message 39');
    expect(text).not.toContain('message 0 ');
  });

  it('includes the summary and the items mentioned', () => {
    const text = renderHistory(
      conv(2, {
        summary: 'User is tracking a visa application.',
        refs: [{ source: 'mail', id: 'm1', title: 'Visa paperwork', date: '2026-11-09' }],
      }),
      900,
    );
    expect(text).toContain('Earlier in this conversation: User is tracking a visa application.');
    expect(text).toContain('mail id=m1 | 2026-11-09 | Visa paperwork');
  });

  it('gives longer contexts a bigger history budget', () => {
    expect(historyBudget(4096)).toBe(901);
    expect(historyBudget(8192)).toBe(1802);
    expect(historyBudget(16384)).toBe(3604);
    expect(historyBudget(32768)).toBe(6000);
  });
});

describe('items mentioned', () => {
  it('collects ids from search results and from tool output', () => {
    const steps: TraceStep[] = [
      {
        kind: 'tool',
        name: 'search_mail',
        args: {},
        ok: true,
        output: '<data source="gmail search">\nid=abc123 | 2026-11-10 09:30 | from: Sam <sam@x.test> | Visa paperwork\n  attached\n</data>',
      },
      {
        kind: 'tool',
        name: 'read_event',
        args: { id: 'ev9' },
        ok: true,
        output: '<data source="calendar event">\nid=ev9\nTitle: Standup\nDate: 2026-11-12 09:30\n\n09:30–09:45\n</data>',
      },
      { kind: 'tool', name: 'search_docs', args: {}, ok: false, output: 'ERROR: x' },
    ];
    const refs = extractRefs(
      [{ source: 'docs', id: 'd1', title: 'Contract', text: '', date: Date.UTC(2026, 9, 1), score: 1, origin: 'index' }],
      steps,
    );
    expect(refs).toEqual([
      { source: 'docs', id: 'd1', title: 'Contract', date: '2026-10-01' },
      { source: 'mail', id: 'abc123', title: 'Visa paperwork', date: '2026-11-10' },
      { source: 'calendar', id: 'ev9', title: 'Standup', date: '2026-11-12' },
    ]);
  });

  it('keeps the newest first, without duplicates, capped at 10', () => {
    const r = (id: string) => ({ source: 'mail' as const, id, title: id, date: '2026-01-01' });
    const merged = mergeRefs([r('new'), r('a')], Array.from({ length: 12 }, (_, i) => r(i === 0 ? 'a' : `o${i}`)));
    expect(merged[0]?.id).toBe('new');
    expect(merged.filter((x) => x.id === 'a')).toHaveLength(1);
    expect(merged).toHaveLength(10);
  });
});

describe('follow-up detection', () => {
  it.each([
    ['what did she say after that?', true],
    ['and the second one?', true],
    ['summarize it', true],
    ["What's on my calendar tomorrow afternoon for the planning meeting?", false],
    ['Find the contract with the 30-day notice period from the landlord', false],
  ])('%s → %s', (q, expected) => {
    expect(needsContext(q)).toBe(expected);
  });
});

function scriptedModel(chat: ChatResult[], json: unknown[] = []) {
  const seen: ChatMessage[][] = [];
  const jsonSeen: ChatMessage[][] = [];
  const model: ChatModel = {
    async chat(messages) {
      seen.push(messages);
      return chat.shift() ?? { content: 'summary of older turns', toolCalls: [] };
    },
    async chatJson(messages) {
      jsonSeen.push(messages);
      return json.shift();
    },
  };
  return { model, seen, jsonSeen };
}

describe('compaction', () => {
  it('folds the oldest turns into the summary when over budget', async () => {
    const { model, seen } = scriptedModel([{ content: 'User asked about visas.', toolCalls: [] }]);
    const out = await compact(conv(30), model, 900);
    expect(out.summary).toBe('User asked about visas.');
    expect(out.turns.length).toBeLessThan(30);
    expect(out.turns.length).toBeGreaterThanOrEqual(4);
    expect(out.turns.at(-1)?.text).toContain('message 29');
    expect(seen[0]?.[1]?.content).toContain('message 0');
  });

  it('leaves a short conversation alone', async () => {
    const { model, seen } = scriptedModel([]);
    const c = conv(4);
    expect(await compact(c, model, 900)).toBe(c);
    expect(seen).toHaveLength(0);
  });

  it('still drops old turns when the summary call fails', async () => {
    const model: ChatModel = {
      chat: async () => {
        throw new Error('model busy');
      },
    };
    const out = await compact(conv(30, { summary: 'old summary' }), model, 900);
    expect(out.summary).toBe('old summary');
    expect(out.turns.length).toBeLessThan(30);
  });
});

describe('Agent with memory', () => {
  const NOW = new Date(2026, 10, 11, 10, 0).getTime();
  const mail: Item = { source: 'mail', id: 'm1', title: 'Visa paperwork', text: 'from Priya: forms attached', date: NOW - 86_400_000, from: 'Priya' };
  const services: Services = {
    mail: { search: async () => [mail], read: async () => mail },
    docs: { search: async () => [], read: async () => mail },
    calendar: { listEvents: async () => [], readEvent: async () => mail },
  };
  const retriever = new HybridRetriever({
    index: new InMemoryVectorIndex(),
    embedder: { embed: async (t) => t.map(() => [1]) },
    live: { mail: async () => [mail] },
    coverage: async () => ({}),
  });

  it('rewrites a follow-up from the conversation and routes on the rewrite', async () => {
    const kv = new MemoryKV();
    const { model, seen, jsonSeen } = scriptedModel(
      [
        { content: 'Priya sent the visa forms [1].', toolCalls: [] },
        { content: 'She said the forms are attached [1].', toolCalls: [] },
      ],
      [
        { relevant: true },
        { standalone: 'What did Priya say in the visa paperwork email?' },
        { relevant: true },
      ],
    );
    const agent = new Agent({ model, retriever, services, loadCoverage: async () => ({}), kv, clock: { now: () => NOW } });

    const first = await agent.ask('find the thread about the visa paperwork');
    expect(first.path[0]).toBe('contextualize');
    const second = await agent.ask('what did she say?');

    expect(second.standalone).toBe('What did Priya say in the visa paperwork email?');
    expect(second.route.route).toBe('fuzzy');
    // The rewrite saw the first exchange, including the item id.
    const rewritePrompt = jsonSeen[1]?.[1]?.content ?? '';
    expect(rewritePrompt).toContain('find the thread about the visa paperwork');
    expect(rewritePrompt).toContain('mail id=m1');
    // The answer step saw the conversation too.
    expect(seen.at(-1)?.[1]?.content).toContain('conversation so far');

    const history = await agent.history();
    expect(history.map((t) => t.role)).toEqual(['user', 'assistant', 'user', 'assistant']);
  });

  it('persists the conversation and starts fresh on New chat', async () => {
    const kv = new MemoryKV();
    const make = () =>
      new Agent({ model: scriptedModel([]).model, retriever, services, loadCoverage: async () => ({}), kv, clock: { now: () => NOW } });
    await make().ask('Draft a reply to Sam');
    expect(await make().history()).toHaveLength(2); // survives a restart
    const agent = make();
    await agent.newConversation();
    expect(await agent.history()).toHaveLength(0);
    expect(await new ConversationStore(kv).current(NOW)).toMatchObject({ turns: [] });
  });
});

describe('context length', () => {
  const GB = 1024 ** 3;
  const m = (id: string) => findModel(id) as NonNullable<ReturnType<typeof findModel>>;

  it('offers options up to the model limit', () => {
    expect(contextOptions(m('qwen3-4b-instruct-2507-q4km'))).toEqual([4096, 8192, 16384, 32768]);
  });

  it('picks the largest context that fits the phone', () => {
    expect(autoContext(m('qwen3-4b-instruct-2507-q4km'), 7.5 * GB)).toBe(16384); // iPhone 17
    expect(autoContext(m('qwen3-4b-instruct-2507-q4km'), 11.2 * GB)).toBe(32768); // 12 GB phones
    expect(autoContext(m('qwen3-8b-q4km'), 11.2 * GB)).toBe(16384);
    expect(autoContext(m('qwen3-8b-q4km'), 7.5 * GB)).toBe(4096); // nothing fits: smallest
  });

  it("uses the user's choice when it is a valid option", () => {
    expect(effectiveContext(m('qwen3-4b-instruct-2507-q4km'), 7.5 * GB, 32768)).toBe(32768);
    expect(effectiveContext(m('qwen3-4b-instruct-2507-q4km'), 7.5 * GB, 12345)).toBe(16384);
  });
});
