import type { Tool } from './agent';
import { asData } from './agent';
import type { Services } from './services';
import type { Item, Source, TimeWindow } from './types';

/**
 * One specialist per service, each with 2–3 tools, so a 4B model chooses between few options.
 * Tool names and descriptions say plainly when to use them and when not to: for a small model, this
 * wording drives most of the quality.
 */

export function isoDate(t: number): string {
  const d = new Date(t);
  const p = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

function formatDateTime(t: number): string {
  const d = new Date(t);
  const p = (n: number) => String(n).padStart(2, '0');
  return `${isoDate(t)} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

/** Parses YYYY-MM-DD as local midnight. Returns undefined for anything else. */
export function parseIsoDate(s: unknown): number | undefined {
  if (typeof s !== 'string') return undefined;
  const m = s.trim().match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (!m) return undefined;
  return new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3])).getTime();
}

const DAY = 24 * 60 * 60 * 1000;

/** start/end as YYYY-MM-DD; `end` is inclusive for the model, exclusive internally. */
function windowFromArgs(args: Record<string, unknown>, startKey: string, endKey: string): TimeWindow | null {
  const start = parseIsoDate(args[startKey]);
  const endDay = parseIsoDate(args[endKey]);
  if (start === undefined && endDay === undefined) return null;
  return { start, end: endDay !== undefined ? endDay + DAY : undefined };
}

function toBool(v: unknown): boolean {
  return v === true || v === 'true';
}

function toInt(v: unknown, dflt: number, max: number): number {
  const n = Number(v);
  if (!Number.isFinite(n) || n <= 0) return dflt;
  return Math.min(Math.floor(n), max);
}

/** One line per item; the id comes first so the model can re-fetch it. */
export function formatItems(items: Item[], maxText = 200): string {
  if (items.length === 0) return 'No results.';
  return items
    .map((it) => {
      const from = it.from ? ` | from: ${it.from}` : '';
      const text = it.text.replace(/\s+/g, ' ').slice(0, maxText);
      return `id=${it.id} | ${formatDateTime(it.date)}${from} | ${it.title}\n  ${text}`;
    })
    .join('\n');
}

export function formatFull(it: Item, maxText = 3000): string {
  const from = it.from ? `From: ${it.from}\n` : '';
  const url = it.url ? `Link: ${it.url}\n` : '';
  return `id=${it.id}\nTitle: ${it.title}\n${from}Date: ${formatDateTime(it.date)}\n${url}\n${it.text.slice(0, maxText)}`;
}

function splitQuery(v: unknown): string[] {
  return typeof v === 'string' ? v.split(/\s+/).filter(Boolean) : [];
}

export function mailTools(s: Services): { search: Tool; read: Tool } {
  return {
    search: {
      spec: {
        name: 'search_mail',
        description:
          'Search Gmail by words, sender, date range or unread status. Use for any question about emails. ' +
          'Returns message ids with a short snippet; use read_message to see a full message.',
        parameters: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'A few key words to match. Leave empty to match all mail.' },
            from: { type: 'string', description: 'Sender name or address, only if the user named one.' },
            after: { type: 'string', description: 'First day to include, YYYY-MM-DD.' },
            before: { type: 'string', description: 'Last day to include, YYYY-MM-DD.' },
            unread_only: { type: 'boolean', description: 'true only if the user asked for unread mail.' },
          },
        },
      },
      async run(args) {
        const items = await s.mail.search({
          keywords: splitQuery(args.query),
          from: typeof args.from === 'string' && args.from ? args.from : undefined,
          window: windowFromArgs(args, 'after', 'before'),
          unreadOnly: toBool(args.unread_only),
          max: 8,
        });
        return asData('gmail search', formatItems(items));
      },
    },
    read: {
      spec: {
        name: 'read_message',
        description:
          'Read one email in full by its id. Use before quoting or summarizing a message. Do not use to search.',
        parameters: {
          type: 'object',
          properties: { id: { type: 'string', description: 'Message id from search results.' } },
          required: ['id'],
        },
      },
      async run(args) {
        return asData('gmail message', formatFull(await s.mail.read(String(args.id))));
      },
    },
  };
}

export function docsTools(s: Services): { search: Tool; read: Tool } {
  return {
    search: {
      spec: {
        name: 'search_docs',
        description:
          'Search Google Drive documents by words in their name or text. Use for questions about documents or files. ' +
          'Returns document ids; use read_doc to see the text.',
        parameters: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'A few key words to match.' },
            modified_after: { type: 'string', description: 'Only documents changed on or after this day, YYYY-MM-DD.' },
          },
          required: ['query'],
        },
      },
      async run(args) {
        const items = await s.docs.search({
          keywords: splitQuery(args.query),
          window: windowFromArgs(args, 'modified_after', 'modified_before'),
          max: 8,
        });
        return asData('drive search', formatItems(items));
      },
    },
    read: {
      spec: {
        name: 'read_doc',
        description: 'Read the text of one Drive document by its id. Use before quoting a document. Do not use to search.',
        parameters: {
          type: 'object',
          properties: { id: { type: 'string', description: 'Document id from search results.' } },
          required: ['id'],
        },
      },
      async run(args) {
        return asData('drive document', formatFull(await s.docs.read(String(args.id))));
      },
    },
  };
}

export function calendarTools(s: Services): { list: Tool; read: Tool } {
  return {
    list: {
      spec: {
        name: 'list_events',
        description:
          'List calendar events between two days. Use for any question about meetings, schedule or free time.',
        parameters: {
          type: 'object',
          properties: {
            start: { type: 'string', description: 'First day, YYYY-MM-DD.' },
            end: { type: 'string', description: 'Last day (inclusive), YYYY-MM-DD.' },
            query: { type: 'string', description: 'Optional words to match in the event, such as a person or topic.' },
          },
          required: ['start', 'end'],
        },
      },
      async run(args) {
        const window = windowFromArgs(args, 'start', 'end');
        if (!window?.start || !window.end) throw new Error('start and end must be dates like 2026-11-11');
        if (window.end <= window.start) throw new Error('end must be on or after start');
        const items = await s.calendar.listEvents({
          window,
          query: typeof args.query === 'string' && args.query ? args.query : undefined,
          max: 25,
        });
        return asData('calendar', formatItems(items, 160));
      },
    },
    read: {
      spec: {
        name: 'read_event',
        description: 'Read one calendar event in full (attendees, location, description) by its id.',
        parameters: {
          type: 'object',
          properties: { id: { type: 'string', description: 'Event id from list_events.' } },
          required: ['id'],
        },
      },
      async run(args) {
        return asData('calendar event', formatFull(await s.calendar.readEvent(String(args.id))));
      },
    },
  };
}

/** The tools a specialist for one service gets on the structured path. */
export function specialistTools(source: Source, s: Services): Tool[] {
  switch (source) {
    case 'mail': {
      const t = mailTools(s);
      return [t.search, t.read];
    }
    case 'docs': {
      const t = docsTools(s);
      return [t.search, t.read];
    }
    case 'calendar': {
      const t = calendarTools(s);
      return [t.list, t.read];
    }
  }
}

/** Read-by-id tools only: used on the recall path to re-fetch an item before quoting it. */
export function readTools(sources: Source[], s: Services): Tool[] {
  const out: Tool[] = [];
  if (sources.includes('mail')) out.push(mailTools(s).read);
  if (sources.includes('docs')) out.push(docsTools(s).read);
  if (sources.includes('calendar')) out.push(calendarTools(s).read);
  return out;
}
