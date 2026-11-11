import type { Embedder } from './llm/types';
import type { Item, ScoredItem, Source, TimeWindow } from './types';
import type { VectorIndex } from './vectorIndex';
import { cosine } from './vectorIndex';
import type { Coverage } from './watermark';
import { isWeak, mergeResults, planRetrieval, WEAK_SCORE } from './watermark';

/** A live keyword search against one service, limited to a time window. */
export type LiveSearch = (q: { keywords: string[]; window: TimeWindow | null; max: number }) => Promise<Item[]>;

export interface RetrievalRequest {
  question: string;
  keywords: string[];
  window: TimeWindow | null;
  sources: Source[];
}

export interface RetrievalTraceEntry {
  source: Source;
  coverage?: Coverage;
  indexWindow: TimeWindow | null;
  liveWindows: TimeWindow[];
  indexHits: number;
  liveHits: number;
  fallback: boolean;
  error?: string;
}

export interface RetrievalResult {
  results: ScoredItem[];
  trace: RetrievalTraceEntry[];
}

export interface RetrieverDeps {
  index: VectorIndex;
  embedder: Embedder;
  live: Partial<Record<Source, LiveSearch>>;
  /** What the index covers per source. Missing = nothing indexed yet. */
  coverage: () => Promise<Partial<Record<Source, Coverage>>>;
  /** How many results go to the model. */
  k?: number;
  weakScore?: number;
}

/**
 * Hybrid retrieval with the watermark rule: index for what is older than the last sync, live
 * fetch for what is newer, then merge and rerank both by similarity to the question, so a hit
 * scores the same whichever side of the watermark it came from.
 */
export class HybridRetriever {
  private k: number;
  private weakScore: number;

  constructor(private deps: RetrieverDeps) {
    this.k = deps.k ?? 6;
    this.weakScore = deps.weakScore ?? WEAK_SCORE;
  }

  async retrieve(req: RetrievalRequest): Promise<RetrievalResult> {
    const [queryVec] = await this.deps.embedder.embed([req.question], 'query');
    if (!queryVec) throw new Error('Embedder returned no vector for the question.');
    const coverage = await this.deps.coverage();

    const perSource = await Promise.all(
      req.sources.map((source) => this.retrieveSource(source, queryVec, req, coverage[source])),
    );
    return {
      results: mergeResults(
        perSource.map((p) => p.results),
        this.k,
      ),
      trace: perSource.map((p) => p.trace),
    };
  }

  private async retrieveSource(
    source: Source,
    queryVec: number[],
    req: RetrievalRequest,
    coverage: Coverage | undefined,
  ): Promise<{ results: ScoredItem[]; trace: RetrievalTraceEntry }> {
    const plan = planRetrieval(req.window, coverage);
    const live = this.deps.live[source];
    const trace: RetrievalTraceEntry = {
      source,
      coverage,
      indexWindow: plan.index,
      liveWindows: live ? plan.live : [],
      indexHits: 0,
      liveHits: 0,
      fallback: false,
    };

    let indexResults: ScoredItem[] = [];
    if (plan.index) {
      const hits = await this.deps.index.search(queryVec, { sources: [source], window: plan.index, k: this.k * 2 });
      indexResults = hits.map(({ chunk, score }) => ({
        source,
        id: chunk.itemId,
        title: chunk.title,
        text: chunk.text,
        date: chunk.date,
        from: chunk.from,
        score,
        origin: 'index' as const,
      }));
      trace.indexHits = indexResults.length;
    }

    let liveResults: ScoredItem[] = [];
    if (plan.live.length > 0 && live) {
      try {
        const fetched = await Promise.all(
          plan.live.map((window) => live({ keywords: req.keywords, window, max: this.k * 2 })),
        );
        liveResults = await this.scoreLive(fetched.flat(), queryVec);
      } catch (e) {
        // A failed live fetch should not throw away good index results.
        trace.error = (e as Error).message;
      }
      trace.liveHits = liveResults.length;
    }

    let merged = mergeResults([indexResults, liveResults], this.k * 2);

    // Weak or empty, and the index had a say: check live across the whole window before giving up.
    if (plan.index && live && isWeak(merged, this.weakScore)) {
      trace.fallback = true;
      try {
        const fallback = await this.scoreLive(await live({ keywords: req.keywords, window: req.window, max: this.k * 2 }), queryVec);
        trace.liveHits += fallback.length;
        merged = mergeResults([merged, fallback], this.k * 2);
      } catch (e) {
        trace.error = (e as Error).message;
      }
    }
    return { results: merged, trace };
  }

  private async scoreLive(items: Item[], queryVec: number[]): Promise<ScoredItem[]> {
    if (items.length === 0) return [];
    const vecs = await this.deps.embedder.embed(items.map((it) => `${it.title}\n${it.text}`), 'document');
    return items.map((it, i) => ({ ...it, score: cosine(queryVec, vecs[i] ?? []), origin: 'live' as const }));
  }
}
