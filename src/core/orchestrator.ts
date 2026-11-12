import type { BaseCheckpointSaver } from '@langchain/langgraph/web';
import type { TraceStep } from './agent';
import type { AgentGraph, AgentStateType, PlanStep } from './graph/agentGraph';
import { buildAgentGraph } from './graph/agentGraph';
import type { ChatModel } from './llm/types';
import type { Conversation, Turn } from './memory';
import { compact, ConversationStore, extractRefs, historyBudget, mergeRefs, renderHistory } from './memory';
import type { HybridRetriever, RetrievalTraceEntry } from './retriever';
import type { RouteDecision } from './router';
import type { KeyValueStore, Services } from './services';
import type { Clock, ScoredItem, Source } from './types';
import { systemClock } from './types';
import type { Coverage } from './watermark';

export interface AgentEvent {
  type: 'node' | 'route' | 'retrieval' | 'step' | 'partial';
  /** For 'partial': the answer streamed so far ('' clears it). */
  text?: string;
  node?: string;
  route?: RouteDecision;
  retrieval?: RetrievalTraceEntry[];
  step?: TraceStep;
}

export interface AgentAnswer {
  answer: string;
  route: RouteDecision;
  retrieval: RetrievalTraceEntry[];
  sources: ScoredItem[];
  steps: TraceStep[];
  plan: PlanStep[];
  /** Graph nodes in the order they ran. */
  path: string[];
  threadId: string;
  question: string;
  /** The question as the graph understood it (follow-ups rewritten to stand alone). */
  standalone: string;
}

export interface AgentDeps {
  model: ChatModel;
  retriever: HybridRetriever;
  services: Services;
  loadCoverage: () => Promise<Partial<Record<Source, Coverage>>>;
  checkpointer?: BaseCheckpointSaver;
  /** Where the list of recent threads is kept, so old checkpoints can be pruned. */
  kv?: KeyValueStore;
  clock?: Clock;
  maxSteps?: number;
  /** Graph super-steps per run; a guard against loops on top of each node's own step limit. */
  recursionLimit?: number;
  keepThreads?: number;
  /** The chat model's context length; sets how much conversation history each turn carries. */
  contextTokens?: number;
}

const THREADS_KEY = 'agent.threads';

/** Runs one question through the LangGraph graph and reports progress as it goes. */
export class Agent {
  readonly graph: AgentGraph;
  private clock: Clock;
  private listeners = new Map<string, (step: TraceStep) => void>();
  private partialListeners = new Map<string, (text: string) => void>();
  /** Conversation memory; present when the agent has a key-value store. */
  readonly conversations: ConversationStore | null;
  /** Folding old turns into the summary runs after an answer is returned; the next turn waits for it. */
  private compacting: Promise<void> = Promise.resolve();

  constructor(private deps: AgentDeps) {
    this.clock = deps.clock ?? systemClock;
    this.conversations = deps.kv ? new ConversationStore(deps.kv) : null;
    this.graph = buildAgentGraph(
      {
        model: deps.model,
        retriever: deps.retriever,
        services: deps.services,
        loadCoverage: deps.loadCoverage,
        now: () => this.clock.now(),
        maxSteps: deps.maxSteps,
        onStep: (threadId, step) => this.listeners.get(threadId)?.(step),
        onPartial: (threadId, text) => this.partialListeners.get(threadId)?.(text),
      },
      deps.checkpointer,
    );
  }

  private get budget(): number {
    return historyBudget(this.deps.contextTokens ?? 8192);
  }

  /** The open conversation's messages, for showing the chat after a restart. */
  async history(): Promise<Turn[]> {
    if (!this.conversations) return [];
    await this.compacting;
    return (await this.conversations.current(this.clock.now())).turns;
  }

  /** Forgets the current conversation and starts a new one. */
  async newConversation(): Promise<void> {
    await this.compacting;
    await this.conversations?.start(this.clock.now());
  }

  async ask(question: string, onEvent?: (e: AgentEvent) => void): Promise<AgentAnswer> {
    await this.compacting;
    const conv = this.conversations ? await this.conversations.current(this.clock.now()) : null;
    const history = renderHistory(conv, this.budget);
    const result = await this.run(question, history, onEvent);
    if (conv) this.remember(conv, question, result);
    return result;
  }

  /** Saves the turn now; compaction (a model call) finishes in the background. */
  private remember(conv: Conversation, question: string, result: AgentAnswer): void {
    const now = this.clock.now();
    const updated: Conversation = {
      ...conv,
      turns: [...conv.turns, { role: 'user', text: question, at: now }, { role: 'assistant', text: result.answer, at: now }],
      refs: mergeRefs(extractRefs(result.sources, result.steps), conv.refs),
    };
    const store = this.conversations as ConversationStore;
    this.compacting = (async () => {
      await store.save(updated);
      const compacted = await compact(updated, this.deps.model, this.budget);
      if (compacted !== updated) await store.save(compacted);
    })().catch(() => undefined);
  }

  private async run(question: string, history: string, onEvent?: (e: AgentEvent) => void): Promise<AgentAnswer> {
    const threadId = `run-${this.clock.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const config = { configurable: { thread_id: threadId }, recursionLimit: this.deps.recursionLimit ?? 25 };
    const path: string[] = [];
    let last: Partial<AgentStateType> = {};

    this.listeners.set(threadId, (step) => onEvent?.({ type: 'step', step }));
    this.partialListeners.set(threadId, (text) => onEvent?.({ type: 'partial', text }));
    try {
      // 'updates' says which node just ran; 'values' carries the full state after it.
      const stream = await this.graph.stream({ question, history }, { ...config, streamMode: ['updates', 'values'] });
      for await (const chunk of stream) {
        const [mode, data] = chunk as [string, Record<string, unknown>];
        if (mode === 'values') {
          last = data as Partial<AgentStateType>;
          continue;
        }
        for (const [node, update] of Object.entries(data)) {
          path.push(node);
          onEvent?.({ type: 'node', node });
          const u = update as Partial<AgentStateType>;
          if (node === 'router' && u.route) onEvent?.({ type: 'route', route: u.route });
          if (node === 'retrieve' && u.retrieval) onEvent?.({ type: 'retrieval', retrieval: u.retrieval });
        }
      }
    } finally {
      this.listeners.delete(threadId);
      this.partialListeners.delete(threadId);
    }
    await this.rememberThread(threadId);

    const state = last as AgentStateType;
    if (!state.route) throw new Error('The agent graph ended without routing the question.');
    return {
      answer: state.answer || 'I could not produce an answer.',
      route: state.route,
      retrieval: state.retrieval ?? [],
      sources: state.route.route === 'fuzzy' ? state.retrieved ?? [] : [],
      steps: state.steps ?? [],
      plan: state.plan ?? [],
      path,
      threadId,
      question,
      standalone: state.standalone || question,
    };
  }

  /** Keeps the newest threads' checkpoints and deletes the rest. */
  private async rememberThread(threadId: string): Promise<void> {
    const { kv, checkpointer } = this.deps;
    if (!kv || !checkpointer) return;
    const keep = this.deps.keepThreads ?? 20;
    const raw = await kv.get(THREADS_KEY);
    const threads = [...(raw ? (JSON.parse(raw) as string[]) : []), threadId];
    const drop = threads.slice(0, Math.max(0, threads.length - keep));
    for (const t of drop) await checkpointer.deleteThread(t);
    await kv.set(THREADS_KEY, JSON.stringify(threads.slice(drop.length)));
  }
}
