import { runToolLoop } from '../src/core/agent';
import type { ChatModel } from '../src/core/llm/types';
import { partialReporter } from '../src/core/llm/onDevice';
import type { Inline } from '../src/core/markdown';
import { inlineText, parseInline, parseMarkdown } from '../src/core/markdown';

describe('parseInline', () => {
  it('handles bold, italic, code and links', () => {
    expect(parseInline('Meet **Priya** at *3pm* in `Room 4`, see [invite](https://x.test/a).')).toEqual([
      { type: 'text', text: 'Meet ' },
      { type: 'bold', children: [{ type: 'text', text: 'Priya' }] },
      { type: 'text', text: ' at ' },
      { type: 'italic', children: [{ type: 'text', text: '3pm' }] },
      { type: 'text', text: ' in ' },
      { type: 'code', text: 'Room 4' },
      { type: 'text', text: ', see ' },
      { type: 'link', href: 'https://x.test/a', children: [{ type: 'text', text: 'invite' }] },
      { type: 'text', text: '.' },
    ]);
  });

  it('keeps unclosed markers as text while streaming', () => {
    expect(inlineText(parseInline('The **visa forms'))).toBe('The **visa forms');
    expect(parseInline('a `b')).toEqual([{ type: 'text', text: 'a `b' }]);
  });

  it('does not italicize snake_case or a lone asterisk', () => {
    expect(parseInline('file_name_v2 and 2 * 3')).toEqual([{ type: 'text', text: 'file_name_v2 and 2 * 3' }]);
  });

  it('links bare URLs without trailing punctuation', () => {
    const nodes = parseInline('Open https://mail.google.com/x.');
    expect(nodes[1]).toMatchObject({ type: 'link', href: 'https://mail.google.com/x' });
    expect((nodes[2] as Extract<Inline, { type: 'text' }>).text).toBe('.');
  });

  it('only links http(s) and mailto', () => {
    expect(parseInline('[x](javascript:alert(1))')[0]).toMatchObject({ type: 'text' });
  });
});

describe('parseMarkdown', () => {
  it('parses a typical answer', () => {
    const blocks = parseMarkdown(
      'You have **3 events** tomorrow:\n\n- **09:30** Standup\n- **13:00** Lunch with Sam\n  (at Café Rio)\n\n1. first\n2. second\n\n> quoted\n\n---\n\n```\ncode\n```',
    );
    expect(blocks.map((b) => b.type)).toEqual(['paragraph', 'list', 'list', 'quote', 'rule', 'code']);
    const list = blocks[1] as Extract<(typeof blocks)[number], { type: 'list' }>;
    expect(list.ordered).toBe(false);
    expect(list.items).toHaveLength(2);
    expect(inlineText(list.items[1]?.content ?? [])).toBe('13:00 Lunch with Sam (at Café Rio)');
    const ordered = blocks[2] as Extract<(typeof blocks)[number], { type: 'list' }>;
    expect(ordered.items.map((i) => i.number)).toEqual([1, 2]);
  });

  it('nests indented items under their parent list', () => {
    const [list] = parseMarkdown('1. Mail\n   - from Sam\n2. Docs');
    expect(list).toMatchObject({ type: 'list', ordered: true });
    expect((list as Extract<typeof list, { type: 'list' }>).items.map((i) => i.depth)).toEqual([0, 1, 0]);
  });

  it('shows an unclosed code fence as code so far', () => {
    expect(parseMarkdown('Here:\n```\nline one')).toEqual([
      { type: 'paragraph', content: [{ type: 'text', text: 'Here:' }] },
      { type: 'code', text: 'line one' },
    ]);
  });

  it('caps heading levels at 3 and joins wrapped paragraph lines', () => {
    expect(parseMarkdown('#### Small\nline one\nline two')).toEqual([
      { type: 'heading', level: 3, content: [{ type: 'text', text: 'Small' }] },
      { type: 'paragraph', content: [{ type: 'text', text: 'line one line two' }] },
    ]);
  });
});

describe('streaming', () => {
  it('reports the parsed answer so far and never tool-call text', () => {
    const seen: string[] = [];
    const report = partialReporter((t) => seen.push(t));
    report({ content: 'You' });
    report({ content: 'You have' });
    report({ content: 'You have' }); // unchanged: not repeated
    report({ content: '<think>hmm' }); // unclosed reasoning is hidden
    expect(seen).toEqual(['You', 'You have', '']);

    const tool: string[] = [];
    const r2 = partialReporter((t) => tool.push(t));
    r2({ content: 'Let me check' });
    r2({ content: 'Let me check', tool_calls: [{}] });
    r2({ content: 'more' }); // ignored once the turn is a tool call
    expect(tool).toEqual(['Let me check', '']);
  });

  it('falls back to raw text, holding back tool-call JSON', () => {
    const seen: string[] = [];
    const report = partialReporter((t) => seen.push(t));
    report({ accumulated_text: '{"name": "list_events"' });
    report({ accumulated_text: '<tool_call>{' });
    expect(seen).toEqual([]);
    const ok = partialReporter((t) => seen.push(t));
    ok({ accumulated_text: 'Hello' });
    expect(seen).toEqual(['Hello']);
  });

  it('streams the answer turn and clears what a tool-call turn showed', async () => {
    const partials: string[] = [];
    let turn = 0;
    const model: ChatModel = {
      async chat(_m, _tools, opts) {
        turn++;
        if (turn === 1) {
          opts?.onPartial?.('Let me look');
          return { content: '', toolCalls: [{ name: 'echo', arguments: {} }] };
        }
        opts?.onPartial?.('Found');
        opts?.onPartial?.('Found it.');
        return { content: 'Found it.', toolCalls: [] };
      },
    };
    const r = await runToolLoop({
      model,
      system: 's',
      user: 'u',
      tools: [{ spec: { name: 'echo', description: '', parameters: { type: 'object', properties: {} } }, run: async () => 'ok' }],
      onPartial: (t) => partials.push(t),
    });
    expect(r.answer).toBe('Found it.');
    expect(partials).toEqual(['Let me look', '', 'Found', 'Found it.']);
  });
});
