import type { Chunk, Item } from './types';

/** The model receives a handful of chunks, never whole documents. */
export const CHUNK_CHARS = 900;
export const CHUNK_OVERLAP = 150;

export function splitText(text: string, size = CHUNK_CHARS, overlap = CHUNK_OVERLAP): string[] {
  const clean = text.replace(/\r/g, '').replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim();
  if (clean.length <= size) return clean ? [clean] : [];
  const out: string[] = [];
  let start = 0;
  while (start < clean.length) {
    let end = Math.min(start + size, clean.length);
    if (end < clean.length) {
      // Prefer to break at a paragraph, then a sentence, then a space.
      const window = clean.slice(start, end);
      const cut = Math.max(window.lastIndexOf('\n\n'), window.lastIndexOf('. '), window.lastIndexOf(' '));
      if (cut > size * 0.5) end = start + cut + 1;
    }
    out.push(clean.slice(start, end).trim());
    if (end >= clean.length) break;
    start = Math.max(end - overlap, start + 1);
  }
  return out.filter(Boolean);
}

/** A short header in every chunk so a chunk read on its own still says what it is. */
export function chunkHeader(item: Pick<Item, 'source' | 'title' | 'from' | 'date'>): string {
  const date = new Date(item.date).toISOString().slice(0, 10);
  const from = item.from ? ` | from ${item.from}` : '';
  return `[${item.source}] ${item.title}${from} | ${date}`;
}

export function chunkItem(item: Item, body: string): Omit<Chunk, 'embedding'>[] {
  const header = chunkHeader(item);
  return splitText(body || item.text).map((text, n) => ({
    id: `${item.id}#${n}`,
    itemId: item.id,
    source: item.source,
    title: item.title,
    from: item.from,
    date: item.date,
    text: `${header}\n${text}`,
  }));
}
