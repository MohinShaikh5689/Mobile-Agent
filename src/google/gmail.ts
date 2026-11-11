import { decodeBase64Url, htmlToText } from '../core/encoding';
import type { MailSearch, MailService } from '../core/services';
import type { Item } from '../core/types';
import type { TokenProvider } from './http';
import { googleFetch, mapLimit, qs } from './http';

const BASE = 'https://gmail.googleapis.com/gmail/v1/users/me';
const BODY_CHARS = 6000;

interface Header {
  name: string;
  value: string;
}
interface Part {
  mimeType?: string;
  body?: { data?: string };
  parts?: Part[];
}
interface GmailMessage {
  id: string;
  threadId: string;
  internalDate?: string;
  snippet?: string;
  labelIds?: string[];
  payload?: Part & { headers?: Header[] };
}

function header(m: GmailMessage, name: string): string {
  return m.payload?.headers?.find((h) => h.name.toLowerCase() === name.toLowerCase())?.value ?? '';
}

function findPart(p: Part | undefined, mime: string): Part | undefined {
  if (!p) return undefined;
  if (p.mimeType === mime && p.body?.data) return p;
  for (const c of p.parts ?? []) {
    const f = findPart(c, mime);
    if (f) return f;
  }
  return undefined;
}

export function messageBody(m: GmailMessage): string {
  const plain = findPart(m.payload, 'text/plain');
  if (plain?.body?.data) return decodeBase64Url(plain.body.data);
  const html = findPart(m.payload, 'text/html');
  if (html?.body?.data) return htmlToText(decodeBase64Url(html.body.data));
  return m.snippet ?? '';
}

function toItem(m: GmailMessage, text: string): Item {
  const unread = m.labelIds?.includes('UNREAD') ? ' (unread)' : '';
  return {
    source: 'mail',
    id: m.id,
    title: (header(m, 'Subject') || '(no subject)') + unread,
    from: header(m, 'From'),
    date: Number(m.internalDate ?? 0),
    text,
    url: `https://mail.google.com/mail/u/0/#all/${m.threadId}`,
  };
}

/** Builds a Gmail search string. Several keywords are OR-ed ({a b}), since all-must-match is too strict. */
export function buildGmailQuery(opts: MailSearch): string {
  const q: string[] = [];
  const kw = (opts.keywords ?? []).map((k) => k.replace(/["{}()]/g, '')).filter(Boolean);
  if (kw.length === 1) q.push(kw[0] as string);
  else if (kw.length > 1) q.push(`{${kw.join(' ')}}`);
  if (opts.from) {
    const from = opts.from.replace(/"/g, '').trim();
    q.push(/\s/.test(from) ? `from:"${from}"` : `from:${from}`);
  }
  // Gmail accepts epoch seconds in after:/before:, which avoids time-zone rounding.
  if (opts.window?.start !== undefined) q.push(`after:${Math.floor(opts.window.start / 1000)}`);
  if (opts.window?.end !== undefined) q.push(`before:${Math.floor(opts.window.end / 1000)}`);
  if (opts.unreadOnly) q.push('is:unread');
  q.push('-in:spam -in:trash');
  return q.join(' ');
}

export class GmailClient implements MailService {
  constructor(private getToken: TokenProvider) {}

  async listIds(q: string, max: number, pageToken?: string): Promise<{ ids: string[]; nextPageToken?: string }> {
    const res = await googleFetch<{ messages?: { id: string }[]; nextPageToken?: string }>(
      this.getToken,
      `${BASE}/messages?${qs({ q, maxResults: Math.min(max, 500), pageToken })}`,
    );
    return { ids: (res.messages ?? []).map((m) => m.id), nextPageToken: res.nextPageToken };
  }

  async getMetadata(id: string): Promise<Item> {
    const m = await googleFetch<GmailMessage>(
      this.getToken,
      `${BASE}/messages/${id}?${qs({ format: 'metadata', metadataHeaders: ['From', 'Subject', 'Date'] })}`,
    );
    return toItem(m, m.snippet ?? '');
  }

  async search(opts: MailSearch): Promise<Item[]> {
    const { ids } = await this.listIds(buildGmailQuery(opts), opts.max ?? 8);
    return mapLimit(ids, 4, (id) => this.getMetadata(id));
  }

  async read(id: string): Promise<Item> {
    const m = await googleFetch<GmailMessage>(this.getToken, `${BASE}/messages/${id}?${qs({ format: 'full' })}`);
    return toItem(m, messageBody(m).slice(0, BODY_CHARS));
  }

  async profileHistoryId(): Promise<string> {
    const p = await googleFetch<{ historyId: string }>(this.getToken, `${BASE}/profile`);
    return p.historyId;
  }

  /** Changes since a history id. Throws GoogleApiError 404 when the id is too old; then resync. */
  async history(startHistoryId: string): Promise<{ added: string[]; deleted: string[]; historyId: string }> {
    const added = new Set<string>();
    const deleted = new Set<string>();
    let pageToken: string | undefined;
    let historyId = startHistoryId;
    do {
      const res = await googleFetch<{
        history?: { messagesAdded?: { message: { id: string } }[]; messagesDeleted?: { message: { id: string } }[] }[];
        historyId: string;
        nextPageToken?: string;
      }>(
        this.getToken,
        `${BASE}/history?${qs({ startHistoryId, historyTypes: ['messageAdded', 'messageDeleted'], pageToken })}`,
      );
      for (const h of res.history ?? []) {
        for (const a of h.messagesAdded ?? []) added.add(a.message.id);
        for (const d of h.messagesDeleted ?? []) {
          deleted.add(d.message.id);
          added.delete(d.message.id);
        }
      }
      historyId = res.historyId;
      pageToken = res.nextPageToken;
    } while (pageToken);
    return { added: [...added], deleted: [...deleted], historyId };
  }
}
