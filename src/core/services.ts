import type { CreateEventAction, EmailAction, UpdateEventAction } from './actions';
import type { Item, TimeWindow } from './types';

/**
 * What the agent needs from each Google service. src/google implements these against the real
 * APIs; tests use fakes. Every result is short and trimmed and carries the item ID.
 */

export interface MailSearch {
  keywords?: string[];
  window?: TimeWindow | null;
  from?: string;
  unreadOnly?: boolean;
  max?: number;
}

export interface MailService {
  search(opts: MailSearch): Promise<Item[]>;
  /** Full (trimmed) message body, fetched fresh by ID. */
  read(id: string): Promise<Item>;
}

export interface DocsSearch {
  keywords?: string[];
  window?: TimeWindow | null;
  max?: number;
}

export interface DocsService {
  search(opts: DocsSearch): Promise<Item[]>;
  read(id: string): Promise<Item>;
}

export interface CalendarService {
  listEvents(opts: { window: TimeWindow; query?: string; max?: number }): Promise<Item[]>;
  readEvent(id: string): Promise<Item>;
}

export interface WriteResult {
  id: string;
  url?: string;
}

/**
 * Write access, used only by the execute step after the user approved a proposed action.
 * Requires the gmail.compose and calendar.events scopes.
 */
export interface WriteService {
  /** Headers of a message being replied to, re-fetched right before writing. */
  replyHeaders(messageId: string): Promise<{ threadId: string; messageId?: string; references?: string; subject: string }>;
  createDraft(a: EmailAction, reply?: { threadId: string; messageId?: string; references?: string }): Promise<WriteResult>;
  sendEmail(a: EmailAction, reply?: { threadId: string; messageId?: string; references?: string }): Promise<WriteResult>;
  createEvent(a: CreateEventAction): Promise<WriteResult>;
  updateEvent(a: UpdateEventAction): Promise<WriteResult>;
}

export interface Services {
  mail: MailService;
  docs: DocsService;
  calendar: CalendarService;
  /** Absent when write access is not set up. */
  write?: WriteService;
}

export interface KeyValueStore {
  get(key: string): Promise<string | null>;
  set(key: string, value: string): Promise<void>;
  delete(key: string): Promise<void>;
}
