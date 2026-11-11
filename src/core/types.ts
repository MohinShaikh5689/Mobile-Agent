/** Shared types for the framework-free core. Nothing in src/core imports React Native. */

export type Source = 'mail' | 'docs' | 'calendar';

/** A time range in epoch milliseconds. A missing bound means "unbounded on that side". */
export interface TimeWindow {
  start?: number;
  end?: number;
}

/** One trimmed item returned by a service, always carrying its ID so it can be re-fetched. */
export interface Item {
  source: Source;
  id: string;
  title: string;
  /** Short text: a snippet, or a chunk of the body. */
  text: string;
  /** Epoch ms: message date, doc modified time, or event start. */
  date: number;
  from?: string;
  url?: string;
}

/** An item plus a relevance score, as handed to the model. */
export interface ScoredItem extends Item {
  score: number;
  /** Where the hit came from, for tracing and the "same answer either side of the watermark" check. */
  origin: 'index' | 'live';
}

/** A stored, embedded piece of an item in the local index. */
export interface Chunk {
  /** `${itemId}#${n}` */
  id: string;
  itemId: string;
  source: Source;
  title: string;
  from?: string;
  date: number;
  text: string;
  embedding: number[];
}

export interface Clock {
  now(): number;
}

export const systemClock: Clock = { now: () => Date.now() };
