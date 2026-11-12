/**
 * A small Markdown parser for chat answers: headings, paragraphs, bullet and numbered lists
 * (one level of nesting), block quotes, fenced code, rules; inline bold, italic, code and links.
 *
 * It must cope with half-written Markdown while an answer streams in, so anything unclosed
 * (a lone `**`, an open code fence) renders as plain text or as code-so-far, never breaks.
 */

export type Inline =
  | { type: 'text'; text: string }
  | { type: 'bold'; children: Inline[] }
  | { type: 'italic'; children: Inline[] }
  | { type: 'code'; text: string }
  | { type: 'link'; href: string; children: Inline[] };

export interface ListItem {
  depth: 0 | 1;
  /** The number shown for ordered items. */
  number?: number;
  content: Inline[];
}

export type Block =
  | { type: 'heading'; level: 1 | 2 | 3; content: Inline[] }
  | { type: 'paragraph'; content: Inline[] }
  | { type: 'list'; ordered: boolean; items: ListItem[] }
  | { type: 'quote'; content: Inline[] }
  | { type: 'code'; text: string }
  | { type: 'rule' };

const BULLET = /^(\s*)[-*+•]\s+(.*)$/;
const ORDERED = /^(\s*)(\d{1,3})[.)]\s+(.*)$/;
const HEADING = /^(#{1,6})\s+(.*?)\s*#*\s*$/;
const FENCE = /^\s*```/;
const RULE = /^\s*([-*_])(\s*\1){2,}\s*$/;
const QUOTE = /^\s*>\s?(.*)$/;

export function parseMarkdown(src: string): Block[] {
  const lines = src.replace(/\r\n?/g, '\n').split('\n');
  const blocks: Block[] = [];
  let para: string[] = [];

  const flushPara = () => {
    if (para.length) blocks.push({ type: 'paragraph', content: parseInline(para.join(' ').trim()) });
    para = [];
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i] as string;

    if (FENCE.test(line)) {
      flushPara();
      const code: string[] = [];
      i++;
      while (i < lines.length && !FENCE.test(lines[i] as string)) code.push(lines[i++] as string);
      // An unclosed fence (still streaming) shows the code so far.
      blocks.push({ type: 'code', text: code.join('\n') });
      continue;
    }
    if (!line.trim()) {
      flushPara();
      continue;
    }
    if (RULE.test(line)) {
      flushPara();
      blocks.push({ type: 'rule' });
      continue;
    }
    let m = line.match(HEADING);
    if (m) {
      flushPara();
      const level = Math.min(3, (m[1] as string).length) as 1 | 2 | 3;
      blocks.push({ type: 'heading', level, content: parseInline(m[2] as string) });
      continue;
    }
    m = line.match(QUOTE);
    if (m) {
      flushPara();
      const prev = blocks[blocks.length - 1];
      const content = parseInline(m[1] as string);
      if (prev?.type === 'quote') prev.content.push({ type: 'text', text: ' ' }, ...content);
      else blocks.push({ type: 'quote', content });
      continue;
    }
    const bullet = line.match(BULLET);
    const ordered = bullet ? null : line.match(ORDERED);
    if (bullet || ordered) {
      flushPara();
      const indent = ((bullet ?? ordered) as RegExpMatchArray)[1]?.length ?? 0;
      const depth: 0 | 1 = indent >= 2 ? 1 : 0;
      const isOrdered = Boolean(ordered);
      const text = bullet ? (bullet[2] as string) : ((ordered as RegExpMatchArray)[3] as string);
      const item: ListItem = { depth, content: parseInline(text) };
      if (ordered) item.number = Number(ordered[2]);
      const prev = blocks[blocks.length - 1];
      // Nested items join the list above them whatever their own marker.
      if (prev?.type === 'list' && (prev.ordered === isOrdered || depth === 1)) prev.items.push(item);
      else blocks.push({ type: 'list', ordered: isOrdered, items: [item] });
      continue;
    }
    // A line indented under a list item continues that item.
    const prev = blocks[blocks.length - 1];
    if (para.length === 0 && prev?.type === 'list' && /^\s{2,}\S/.test(line)) {
      const last = prev.items[prev.items.length - 1] as ListItem;
      last.content.push({ type: 'text', text: ' ' }, ...parseInline(line.trim()));
      continue;
    }
    para.push(line.trim());
  }
  flushPara();
  return blocks;
}

/** Inline Markdown. Unmatched markers stay as literal text. */
export function parseInline(src: string): Inline[] {
  const out: Inline[] = [];
  let text = '';
  const flush = () => {
    if (text) out.push({ type: 'text', text });
    text = '';
  };

  let i = 0;
  while (i < src.length) {
    const ch = src[i] as string;
    const rest = src.slice(i);

    // Escaped character.
    if (ch === '\\' && i + 1 < src.length && /[\\`*_[\]()#>-]/.test(src[i + 1] as string)) {
      text += src[i + 1];
      i += 2;
      continue;
    }
    // `code`
    if (ch === '`') {
      const end = src.indexOf('`', i + 1);
      if (end > i + 1) {
        flush();
        out.push({ type: 'code', text: src.slice(i + 1, end) });
        i = end + 1;
        continue;
      }
    }
    // **bold** or __bold__
    if (rest.startsWith('**') || rest.startsWith('__')) {
      const marker = rest.slice(0, 2);
      const end = src.indexOf(marker, i + 2);
      if (end > i + 2) {
        flush();
        out.push({ type: 'bold', children: parseInline(src.slice(i + 2, end)) });
        i = end + 2;
        continue;
      }
    }
    // *italic* or _italic_ (not inside words for _, and not "* " list-like spacing)
    if ((ch === '*' || ch === '_') && src[i + 1] !== ' ' && src[i + 1] !== ch) {
      const wordBefore = i > 0 && /\w/.test(src[i - 1] as string);
      if (!(ch === '_' && wordBefore)) {
        let end = src.indexOf(ch, i + 1);
        while (end > -1 && src[end + 1] === ch) end = src.indexOf(ch, end + 2);
        if (end > i + 1 && src[end - 1] !== ' ' && !(ch === '_' && /\w/.test(src[end + 1] ?? ''))) {
          flush();
          out.push({ type: 'italic', children: parseInline(src.slice(i + 1, end)) });
          i = end + 1;
          continue;
        }
      }
    }
    // [label](https://…)
    if (ch === '[') {
      const m = rest.match(/^\[([^\]]+)\]\(([^)\s]+)\)/);
      if (m && /^(https?:|mailto:)/i.test(m[2] as string)) {
        flush();
        out.push({ type: 'link', href: m[2] as string, children: parseInline(m[1] as string) });
        i += m[0].length;
        continue;
      }
    }
    // Bare URLs.
    if (ch === 'h' && /^https?:\/\/\S+/.test(rest) && (i === 0 || /\s|\(/.test(src[i - 1] as string))) {
      const url = (rest.match(/^https?:\/\/[^\s)]+/) as RegExpMatchArray)[0].replace(/[.,;:!?]+$/, '');
      flush();
      out.push({ type: 'link', href: url, children: [{ type: 'text', text: url }] });
      i += url.length;
      continue;
    }
    text += ch;
    i++;
  }
  flush();
  return out;
}

/** Plain text of inline content (for accessibility labels and tests). */
export function inlineText(nodes: Inline[]): string {
  return nodes
    .map((n) => (n.type === 'text' || n.type === 'code' ? n.text : inlineText(n.children)))
    .join('');
}
