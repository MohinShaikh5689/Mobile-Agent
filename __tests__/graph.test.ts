import { Annotation, END, MemorySaver, START, StateGraph } from '@langchain/langgraph/web';
import { DurableCheckpointSaver } from '../src/core/graph/checkpointer';
import { MAX_REWRITES } from '../src/core/graph/agentGraph';
import type { ChatMessage, ChatModel, ChatResult, ToolSpec } from '../src/core/llm/types';
import { Agent } from '../src/core/orchestrator';
import { HybridRetriever } from '../src/core/retriever';
import type { KeyValueStore, Services } from '../src/core/services';
import type { Item } from '../src/core/types';
import { InMemoryVectorIndex } from '../src/core/vectorIndex';

const NOW = new Date(2026, 10, 11, 10, 0).getTime();
const DAY = 86_400_000;

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

/** Scripted model: separate queues for tool-calling chat and structured (JSON) output. */
function scripted(chat: ChatResult[], json: unknown[] = []) {
  const seen: { messages: ChatMessage[]; tools?: ToolSpec[] }[] = [];
  const jsonSeen: ChatMessage[][] = [];
  const model: ChatModel = {
    async chat(messages, tools) {
      seen.push({ messages, tools });
      return chat.shift() ?? { content: 'done', toolCalls: [] };
    },
    async chatJson(messages) {
      jsonSeen.push(messages);
      if (json.length === 0) throw new Error('no scripted JSON left');
      return json.shift();
    },
  };
  return { model, seen, jsonSeen };
}

const event: Item = { source: 'calendar', id: 'e1', title: 'Standup', text: '09:30–09:45', date: NOW + DAY };
const visaMail: Item = { source: 'mail', id: 'm1', title: 'Visa paperwork', text: 'the visa paperwork is attached', date: NOW - 2 * DAY };

const services: Services = {
  mail: { search: async () => [visaMail], read: async () => visaMail },
  docs: { search: async () => [], read: async () => visaMail },
  calendar: { listEvents: async () => [event], readEvent: async () => event },
};

/** Live mail search that records the keywords it was asked for. */
function makeRetriever(liveCalls: string[][]) {
  return new HybridRetriever({
    index: new InMemoryVectorIndex(),
    embedder: { embed: async (t) => t.map((x) => (x.toLowerCase().includes('visa') ? [1, 0] : [0, 1])) },
    live: {
      mail: async ({ keywords }) => {
        liveCalls.push(keywords);
        return keywords.includes('visa') ? [visaMail] : [];
      },
      docs: async () => [],
    },
    coverage: async () => ({}),
  });
}

function makeAgent(model: ChatModel, extra: Partial<ConstructorParameters<typeof Agent>[0]> = {}, liveCalls: string[][] = []) {
  return new Agent({
    model,
    retriever: makeRetriever(liveCalls),
    services,
    loadCoverage: async () => ({}),
    clock: { now: () => NOW },
    ...extra,
  });
}

describe('agent graph', () => {
  it('calendar: the code fetches the events, the model only answers', async () => {
    const { model, seen } = scripted([{ content: 'You have Standup at 09:30.', toolCalls: [] }]);
    const r = await makeAgent(model).ask("What's on my calendar tomorrow?");
    expect(r.path).toEqual(['contextualize', 'router', 'specialist']);
    // The events were fetched before the model ran, for tomorrow's date.
    expect(r.steps[0]).toMatchObject({ kind: 'tool', name: 'list_events', args: { start: '2026-11-12', end: '2026-11-12' }, ok: true });
    expect(seen[0]?.messages[1]?.content).toContain('Standup');
    // The model may only open one event for details.
    expect(seen[0]?.tools?.map((t) => t.name)).toEqual(['read_event']);
    expect(r.answer).toBe('You have Standup at 09:30.');
  });

  it('calendar: "upcoming" with no date looks at the next 7 days', async () => {
    const { model } = scripted([{ content: 'Standup tomorrow.', toolCalls: [] }]);
    const r = await makeAgent(model).ask('Any upcoming events on my calendar??');
    expect(r.steps[0]).toMatchObject({ name: 'list_events', args: { start: '2026-11-11', end: '2026-11-18' } });
  });

  it('calendar: a permission error is the answer, not a guess', async () => {
    const denied: Services = {
      ...services,
      calendar: {
        listEvents: async () => {
          throw Object.assign(new Error('Google Calendar access was not granted when you signed in.'), { fatal: true });
        },
        readEvent: async () => event,
      },
    };
    const { model, seen } = scripted([]);
    const agent = new Agent({ model, retriever: makeRetriever([]), services: denied, loadCoverage: async () => ({}), clock: { now: () => NOW } });
    const r = await agent.ask("What's on my calendar tomorrow?");
    expect(r.answer).toBe('Google Calendar access was not granted when you signed in.');
    expect(seen).toHaveLength(0);
  });

  it('mail lookups must call a tool before answering', async () => {
    const calls: (string | undefined)[] = [];
    const model: ChatModel = {
      async chat(_m, _tools, opts) {
        calls.push(opts?.toolChoice);
        return calls.length === 1
          ? { content: '', toolCalls: [{ name: 'search_mail', arguments: { unread_only: true } }] }
          : { content: 'One unread email.', toolCalls: [] };
      },
    };
    const r = await makeAgent(model).ask('Any unread mail from today?');
    expect(calls).toEqual(['required', 'auto']);
    expect(r.answer).toBe('One unread email.');
  });

  it('action: no model call', async () => {
    const { model, seen, jsonSeen } = scripted([]);
    const r = await makeAgent(model).ask('Draft a reply to Sam');
    expect(r.path).toEqual(['contextualize', 'router', 'action']);
    expect(seen.length + jsonSeen.length).toBe(0);
  });

  it('fuzzy: retrieve → grade → respond when the grader accepts', async () => {
    const { model, jsonSeen } = scripted([{ content: 'It is attached [1].', toolCalls: [] }], [{ relevant: true }]);
    const r = await makeAgent(model).ask('the thread about the visa paperwork');
    expect(r.path).toEqual(['contextualize', 'router', 'retrieve', 'grade', 'respond']);
    expect(r.sources.map((s) => s.id)).toEqual(['m1']);
    expect(jsonSeen[0]?.[1]?.content).toMatch(/<data source="search results">/);
    expect(r.answer).toBe('It is attached [1].');
  });

  it('fuzzy: the grader rewrites the query once, then answers', async () => {
    const live: string[][] = [];
    const { model } = scripted(
      [{ content: 'Found it [1].', toolCalls: [] }],
      [
        { relevant: false, better_query: 'visa paperwork' },
        { relevant: false, better_query: 'another try' }, // ignored: rewrite budget spent
      ],
    );
    const r = await makeAgent(model, {}, live).ask('the thread about the immigration documents');
    expect(MAX_REWRITES).toBe(1);
    expect(r.path).toEqual(['contextualize', 'router', 'retrieve', 'grade', 'rewrite', 'retrieve', 'grade', 'respond']);
    expect(live.at(-1)).toEqual(['visa', 'paperwork']);
    expect(r.sources.map((s) => s.id)).toEqual(['m1']);
  });

  it('fuzzy: a grader failure does not block the answer', async () => {
    const { model } = scripted([{ content: 'answer', toolCalls: [] }], []);
    const r = await makeAgent(model).ask('the thread about the visa paperwork');
    expect(r.path).toEqual(['contextualize', 'router', 'retrieve', 'grade', 'respond']);
    expect(r.steps.some((s) => s.kind === 'note' && /grader failed/.test(s.message))).toBe(true);
  });

  it('multi_step: planner → one run_step per step → compose', async () => {
    const { model, seen } = scripted(
      [
        { content: 'Standup tomorrow 09:30.', toolCalls: [] },
        { content: 'Sam sent the visa paperwork.', toolCalls: [] },
        { content: 'Combined answer', toolCalls: [] },
      ],
      [{ steps: [{ service: 'calendar', task: 'find the 3pm meeting' }, { service: 'mail', task: 'find the attendees thread' }] }],
    );
    const r = await makeAgent(model).ask('Reschedule my 3pm meeting and draft a note to the attendees');
    expect(r.path).toEqual(['contextualize', 'router', 'planner', 'run_step', 'run_step', 'compose']);
    expect(r.plan.map((p) => p.service)).toEqual(['calendar', 'mail']);
    expect(seen[1]?.messages[1]?.content).toMatch(/Found so far/);
    expect(seen.at(-1)?.messages[1]?.content).toMatch(/Standup tomorrow 09:30\.[\s\S]*Sam sent the visa paperwork\./);
    expect(r.answer).toBe('Combined answer');
  });

  it('multi_step: an invalid plan falls back to one step per service', async () => {
    const { model } = scripted([], [{ steps: [{ service: 'slack', task: 'x' }] }]);
    const r = await makeAgent(model).ask('Reschedule my 3pm meeting and draft a note to the attendees');
    expect(r.plan.map((p) => p.service).sort()).toEqual(['calendar', 'mail']);
  });

  it('reports node events while it runs', async () => {
    const { model } = scripted([{ content: 'ok', toolCalls: [] }]);
    const nodes: string[] = [];
    await makeAgent(model).ask("What's on my calendar tomorrow?", (e) => e.type === 'node' && nodes.push(e.node as string));
    expect(nodes).toEqual(['contextualize', 'router', 'specialist']);
  });
});

describe('DurableCheckpointSaver', () => {
  it('keeps a finished run readable after a restart', async () => {
    const kv = new MemoryKV();
    const { model } = scripted([{ content: 'Standup at 09:30.', toolCalls: [] }]);
    const r = await makeAgent(model, { checkpointer: new DurableCheckpointSaver(kv), kv }).ask("What's on my calendar tomorrow?");

    // "Restart": new saver, new graph, same storage.
    const reopened = makeAgent(scripted([]).model, { checkpointer: new DurableCheckpointSaver(kv) });
    const state = await reopened.graph.getState({ configurable: { thread_id: r.threadId } });
    expect(state.values.answer).toBe('Standup at 09:30.');
    expect(state.next).toEqual([]);
  });

  it('resumes a paused run after a restart (the approval-gate pattern)', async () => {
    const kv = new MemoryKV();
    const S = Annotation.Root({ log: Annotation<string[]>({ reducer: (a, b) => a.concat(b), default: () => [] }) });
    const build = (saver: DurableCheckpointSaver) =>
      new StateGraph(S)
        .addNode('propose', async () => ({ log: ['proposed'] }))
        .addNode('execute', async () => ({ log: ['executed'] }))
        .addEdge(START, 'propose')
        .addEdge('propose', 'execute')
        .addEdge('execute', END)
        .compile({ checkpointer: saver, interruptBefore: ['execute'] });
    const config = { configurable: { thread_id: 't1' } };

    await build(new DurableCheckpointSaver(kv)).invoke({ log: [] }, config);
    const afterRestart = build(new DurableCheckpointSaver(kv));
    expect((await afterRestart.getState(config)).next).toEqual(['execute']);
    const done = await afterRestart.invoke(null, config);
    expect(done.log).toEqual(['proposed', 'executed']);
  });

  it('matches MemorySaver on the same run', async () => {
    const S = Annotation.Root({ n: Annotation<number>({ reducer: (a, b) => a + b, default: () => 0 }) });
    const build = (saver: MemorySaver) =>
      new StateGraph(S)
        .addNode('a', async () => ({ n: 1 }))
        .addNode('b', async () => ({ n: 2 }))
        .addEdge(START, 'a')
        .addEdge('a', 'b')
        .addEdge('b', END)
        .compile({ checkpointer: saver });
    const cfg = { configurable: { thread_id: 'x' } };
    const mem = await build(new MemorySaver()).invoke({ n: 0 }, cfg);
    const dur = await build(new DurableCheckpointSaver(new MemoryKV())).invoke({ n: 0 }, cfg);
    expect(dur).toEqual(mem);
  });

  it('prunes old threads', async () => {
    const kv = new MemoryKV();
    const checkpointer = new DurableCheckpointSaver(kv);
    const agent = makeAgent(scripted([]).model, { checkpointer, kv, keepThreads: 2 });
    const ids: string[] = [];
    for (let i = 0; i < 3; i++) ids.push((await agent.ask('Draft a reply to Sam')).threadId);
    expect(kv.data.has(`checkpoint:${ids[0]}`)).toBe(false);
    expect(kv.data.has(`checkpoint:${ids[2]}`)).toBe(true);
    expect(JSON.parse(kv.data.get('agent.threads') as string)).toEqual(ids.slice(1));
  });
});

describe('streaming through the graph', () => {
  it('streams the final answer as partial events', async () => {
    const model: ChatModel = {
      async chat(_m, _tools, opts) {
        opts?.onPartial?.('- **09:30**');
        opts?.onPartial?.('- **09:30** Standup');
        return { content: '- **09:30** Standup', toolCalls: [] };
      },
    };
    const partials: string[] = [];
    const r = await makeAgent(model).ask("What's on my calendar tomorrow?", (e) => {
      if (e.type === 'partial') partials.push(e.text ?? '');
    });
    expect(partials).toEqual(['- **09:30**', '- **09:30** Standup']);
    expect(r.answer).toBe('- **09:30** Standup');
  });
});
