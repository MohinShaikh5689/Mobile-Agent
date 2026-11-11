import type { TraceStep } from './agent';
import { asData, runToolLoop } from './agent';
import type { ChatModel } from './llm/types';
import type { HybridRetriever, RetrievalTraceEntry } from './retriever';
import type { RouteDecision } from './router';
import { routeRequest } from './router';
import type { Services } from './services';
import { isoDate, readTools, specialistTools } from './specialists';
import type { Clock, ScoredItem, Source } from './types';
import { systemClock } from './types';

export interface AgentEvent {
  type: 'route' | 'retrieval' | 'step';
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
}

export interface AgentDeps {
  model: ChatModel;
  retriever: HybridRetriever;
  services: Services;
  clock?: Clock;
  maxSteps?: number;
}

const SERVICE_NAME: Record<Source, string> = { mail: 'Gmail', docs: 'Google Drive', calendar: 'Google Calendar' };

const WRITE_NOT_YET =
  'Write actions (drafts, invites, event changes) are not built yet. They arrive in phase 4, behind an approval ' +
  'step that shows exactly what will happen before anything is sent. For now I can look things up for you.';

function dateContext(now: number): string {
  const d = new Date(now);
  const weekday = d.toLocaleDateString('en-US', { weekday: 'long' });
  return `Today is ${weekday}, ${isoDate(now)}. Local time ${d.toTimeString().slice(0, 5)}.`;
}

function windowHint(route: RouteDecision): string {
  if (!route.window) return '';
  const { start, end } = route.window;
  const s = start !== undefined ? isoDate(start) : 'the beginning';
  // Windows are end-exclusive; tools take inclusive end days.
  const e = end !== undefined ? isoDate(end - 1) : 'now';
  return ` The user's time range "${route.windowPhrase}" means ${s} to ${e}.`;
}

/**
 * The fixed outer workflow. Code decides the path; the model only makes narrow choices inside it
 * (which tool, which arguments, how to phrase the answer).
 */
export class Agent {
  private clock: Clock;

  constructor(private deps: AgentDeps) {
    this.clock = deps.clock ?? systemClock;
  }

  async ask(question: string, onEvent?: (e: AgentEvent) => void): Promise<AgentAnswer> {
    const now = this.clock.now();
    const route = routeRequest(question, now);
    onEvent?.({ type: 'route', route });
    const onStep = (step: TraceStep) => onEvent?.({ type: 'step', step });

    switch (route.route) {
      case 'action':
        return { answer: WRITE_NOT_YET, route, retrieval: [], sources: [], steps: [] };
      case 'structured':
        return this.structured(question, route, now, onStep);
      case 'fuzzy':
        return this.recall(question, route, now, onEvent, onStep);
      case 'multi_step':
        return this.multiStep(question, route, now, onStep);
    }
  }

  /** Exact lookups go straight to one service's API through its specialist. */
  private async structured(
    question: string,
    route: RouteDecision,
    now: number,
    onStep: (s: TraceStep) => void,
  ): Promise<AgentAnswer> {
    const source = route.services[0] ?? 'mail';
    const res = await runToolLoop({
      model: this.deps.model,
      system:
        `You answer questions about the user's ${SERVICE_NAME[source]}. ${dateContext(now)}${windowHint(route)} ` +
        'Use the tools to look things up; never guess. Answer in a few short sentences or a short list, with dates and names.',
      user: question,
      tools: specialistTools(source, this.deps.services),
      maxSteps: this.deps.maxSteps,
      onStep,
    });
    return { answer: res.answer, route, retrieval: [], sources: [], steps: res.steps };
  }

  /** Fuzzy recall: hybrid retrieval with the watermark rule, then answer from the chunks. */
  private async recall(
    question: string,
    route: RouteDecision,
    now: number,
    onEvent: ((e: AgentEvent) => void) | undefined,
    onStep: (s: TraceStep) => void,
  ): Promise<AgentAnswer> {
    const { results, trace } = await this.deps.retriever.retrieve({
      question,
      keywords: route.keywords,
      window: route.window,
      sources: route.services,
    });
    onEvent?.({ type: 'retrieval', retrieval: trace });

    if (results.length === 0) {
      const where = route.services.map((s) => SERVICE_NAME[s]).join(' or ');
      return {
        answer: `I couldn't find anything about that in ${where}${route.windowPhrase ? ` for ${route.windowPhrase}` : ''}.`,
        route,
        retrieval: trace,
        sources: [],
        steps: [],
      };
    }

    const context = results
      .map((r, i) => `[${i + 1}] id=${r.id} (${r.source}, ${isoDate(r.date)})\n${r.text.slice(0, 700)}`)
      .join('\n\n');

    const res = await runToolLoop({
      model: this.deps.model,
      system:
        `You answer from search results over the user's email and documents. ${dateContext(now)} ` +
        'Use only the results given. If they do not answer the question, say so. Cite results like [1]. ' +
        'Call a read tool only when you need the full text of one result to quote it exactly.',
      user: `${asData('search results', context)}\n\nQuestion: ${question}`,
      tools: readTools(route.services, this.deps.services),
      maxSteps: Math.min(this.deps.maxSteps ?? 5, 3),
      onStep,
    });
    return { answer: res.answer, route, retrieval: trace, sources: results, steps: res.steps };
  }

  /**
   * Phase 1 stand-in for the planner: each service's specialist answers its part in turn, then one
   * call combines the findings. The LangGraph planner replaces this in phase 3.
   */
  private async multiStep(
    question: string,
    route: RouteDecision,
    now: number,
    onStep: (s: TraceStep) => void,
  ): Promise<AgentAnswer> {
    const services = route.services.length > 0 ? route.services : (['mail'] as Source[]);
    const findings: string[] = [];
    const steps: TraceStep[] = [];
    for (const source of services) {
      const res = await runToolLoop({
        model: this.deps.model,
        system:
          `You look things up in the user's ${SERVICE_NAME[source]} as one part of a larger request. ` +
          `${dateContext(now)}${windowHint(route)} Only gather facts from ${SERVICE_NAME[source]}; ` +
          'do not take any action. Report what you found in 2–4 short lines.',
        user: question,
        tools: specialistTools(source, this.deps.services),
        maxSteps: this.deps.maxSteps,
        onStep,
      });
      steps.push(...res.steps);
      findings.push(`${SERVICE_NAME[source]}:\n${res.answer}`);
    }
    const final = await this.deps.model.chat([
      {
        role: 'system',
        content:
          `${dateContext(now)} Combine the findings below into one short answer to the user's request. ` +
          'If the request asks to send, draft, invite or change anything, say what you would propose, and that ' +
          'write actions need the approval step that is not built yet.',
      },
      { role: 'user', content: `${asData('findings', findings.join('\n\n'))}\n\nRequest: ${question}` },
    ]);
    return { answer: final.content, route, retrieval: [], sources: [], steps };
  }
}
