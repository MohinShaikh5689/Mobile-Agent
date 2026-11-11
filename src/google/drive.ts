import type { DocsSearch, DocsService } from '../core/services';
import type { Item } from '../core/types';
import type { TokenProvider } from './http';
import { googleFetch, mapLimit, qs } from './http';

const BASE = 'https://www.googleapis.com/drive/v3';
const GOOGLE_DOC = 'application/vnd.google-apps.document';
const TEXT_CHARS = 20000;
const FIELDS = 'id,name,mimeType,modifiedTime,webViewLink,trashed,owners(displayName)';

interface DriveFile {
  id: string;
  name: string;
  mimeType: string;
  modifiedTime: string;
  webViewLink?: string;
  trashed?: boolean;
  owners?: { displayName: string }[];
}

/** v1 indexes Google Docs and plain-text files; other types are listed by name only. */
export function isReadable(mimeType: string): boolean {
  return mimeType === GOOGLE_DOC || mimeType.startsWith('text/');
}

function toItem(f: DriveFile, text: string): Item {
  return {
    source: 'docs',
    id: f.id,
    title: f.name,
    from: f.owners?.[0]?.displayName,
    date: Date.parse(f.modifiedTime),
    text,
    url: f.webViewLink,
  };
}

function escapeQ(s: string): string {
  return s.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
}

export function buildDriveQuery(opts: DocsSearch): string {
  const q = ['trashed = false', "mimeType != 'application/vnd.google-apps.folder'"];
  const kw = (opts.keywords ?? []).filter(Boolean).slice(0, 6);
  if (kw.length) q.push(`(${kw.map((k) => `fullText contains '${escapeQ(k)}'`).join(' or ')})`);
  if (opts.window?.start !== undefined) q.push(`modifiedTime >= '${new Date(opts.window.start).toISOString()}'`);
  if (opts.window?.end !== undefined) q.push(`modifiedTime < '${new Date(opts.window.end).toISOString()}'`);
  return q.join(' and ');
}

export class DriveClient implements DocsService {
  constructor(private getToken: TokenProvider) {}

  async listFiles(q: string, max: number, pageToken?: string): Promise<{ files: DriveFile[]; nextPageToken?: string }> {
    const res = await googleFetch<{ files?: DriveFile[]; nextPageToken?: string }>(
      this.getToken,
      `${BASE}/files?${qs({ q, pageSize: Math.min(max, 100), orderBy: 'modifiedTime desc', fields: `nextPageToken,files(${FIELDS})`, pageToken })}`,
    );
    return { files: res.files ?? [], nextPageToken: res.nextPageToken };
  }

  async getFile(id: string): Promise<DriveFile> {
    return googleFetch<DriveFile>(this.getToken, `${BASE}/files/${id}?${qs({ fields: FIELDS })}`);
  }

  async fileText(f: DriveFile): Promise<string> {
    if (f.mimeType === GOOGLE_DOC) {
      const text = await googleFetch<string>(
        this.getToken,
        `${BASE}/files/${f.id}/export?${qs({ mimeType: 'text/plain' })}`,
        { asText: true },
      );
      return text.slice(0, TEXT_CHARS);
    }
    if (f.mimeType.startsWith('text/')) {
      const text = await googleFetch<string>(this.getToken, `${BASE}/files/${f.id}?alt=media`, { asText: true });
      return text.slice(0, TEXT_CHARS);
    }
    return '';
  }

  async search(opts: DocsSearch): Promise<Item[]> {
    const { files } = await this.listFiles(buildDriveQuery(opts), opts.max ?? 8);
    // Drive search returns no snippet, so pull the opening text of the top few readable files.
    const withText = files.map((f, i) => ({ f, fetchText: i < 4 && isReadable(f.mimeType) }));
    return mapLimit(withText, 3, async ({ f, fetchText }) =>
      toItem(f, fetchText ? (await this.fileText(f)).slice(0, 1500) : `(${f.mimeType})`),
    );
  }

  async read(id: string): Promise<Item> {
    const f = await this.getFile(id);
    const text = isReadable(f.mimeType) ? await this.fileText(f) : `This file type (${f.mimeType}) cannot be read yet.`;
    return toItem(f, text);
  }

  async readFull(f: DriveFile): Promise<Item> {
    return toItem(f, await this.fileText(f));
  }

  async startPageToken(): Promise<string> {
    const r = await googleFetch<{ startPageToken: string }>(this.getToken, `${BASE}/changes/startPageToken`);
    return r.startPageToken;
  }

  /** Files changed since a page token, plus the token to use next time. */
  async changes(pageToken: string): Promise<{ changed: DriveFile[]; removed: string[]; newToken: string }> {
    const changed: DriveFile[] = [];
    const removed: string[] = [];
    let token: string | undefined = pageToken;
    let newToken = pageToken;
    while (token) {
      const res: {
        changes?: { fileId: string; removed?: boolean; file?: DriveFile }[];
        nextPageToken?: string;
        newStartPageToken?: string;
      } = await googleFetch(
        this.getToken,
        `${BASE}/changes?${qs({ pageToken: token, pageSize: 100, fields: `nextPageToken,newStartPageToken,changes(fileId,removed,file(${FIELDS}))` })}`,
      );
      for (const c of res.changes ?? []) {
        if (c.removed || !c.file || c.file.trashed) removed.push(c.fileId);
        else changed.push(c.file);
      }
      if (res.newStartPageToken) newToken = res.newStartPageToken;
      token = res.nextPageToken;
    }
    return { changed, removed, newToken };
  }
}
