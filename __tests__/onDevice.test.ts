import type { InitFn, LlamaLike } from '../src/core/llm/onDevice';
import { OnDeviceModel, toOaiMessages } from '../src/core/llm/onDevice';
import { MODELS, findModel } from '../src/core/llm/models';

type Completion = Awaited<ReturnType<LlamaLike['completion']>>;

/** A fake llama.rn: records calls, checks nothing runs concurrently, returns scripted output. */
function fakeLlama(script: Completion[] = []) {
  const inits: Record<string, unknown>[] = [];
  const completions: Record<string, unknown>[] = [];
  const embedded: string[] = [];
  let active = 0;
  let maxActive = 0;
  let released = 0;
  const init: InitFn = async (params) => {
    inits.push(params);
    return {
      async completion(p) {
        active++;
        maxActive = Math.max(maxActive, active);
        completions.push(p);
        await new Promise((r) => setTimeout(r, 5));
        active--;
        return script.shift() ?? { text: 'ok', content: 'ok' };
      },
      async embedding(text) {
        active++;
        maxActive = Math.max(maxActive, active);
        embedded.push(text);
        await new Promise((r) => setTimeout(r, 1));
        active--;
        return { embedding: [text.length, 1] };
      },
      async release() {
        released++;
      },
    };
  };
  return { init, inits, completions, embedded, stats: () => ({ maxActive, released }) };
}

const cfg = {
  chatModelPath: '/models/chat.gguf',
  embedModelPath: '/models/embed.gguf',
  chatCtx: 8192,
  embedCtx: 2048,
  queryPrefix: 'search_query: ',
  documentPrefix: 'search_document: ',
};

describe('OnDeviceModel', () => {
  it('maps tool calls from llama.rn and sends tools in OpenAI format', async () => {
    const f = fakeLlama([
      { text: '', content: '', tool_calls: [{ type: 'function', function: { name: 'list_events', arguments: '{"start":"2026-11-12","end":"2026-11-12"}' } }] },
    ]);
    const m = new OnDeviceModel(cfg, f.init);
    const res = await m.chat([{ role: 'user', content: 'tomorrow?' }], [
      { name: 'list_events', description: 'd', parameters: { type: 'object', properties: {} } },
    ]);
    expect(res.toolCalls).toEqual([{ name: 'list_events', arguments: { start: '2026-11-12', end: '2026-11-12' } }]);
    expect(f.completions[0]).toMatchObject({
      jinja: true,
      temperature: 0,
      tool_choice: 'auto',
      tools: [{ type: 'function', function: { name: 'list_events' } }],
    });
    expect(f.inits[0]).toMatchObject({ model: '/models/chat.gguf', n_ctx: 8192 });
  });

  it('survives malformed tool arguments', async () => {
    const f = fakeLlama([{ text: '', tool_calls: [{ function: { name: 'read_message', arguments: '{bad json' } }] }]);
    const res = await new OnDeviceModel(cfg, f.init).chat([{ role: 'user', content: 'x' }], []);
    expect(res.toolCalls).toEqual([{ name: 'read_message', arguments: {} }]);
  });

  it('strips <think> blocks from the answer', async () => {
    const f = fakeLlama([{ text: '<think>hmm</think>Answer', content: '<think>hmm</think>Answer' }]);
    expect((await new OnDeviceModel(cfg, f.init).chat([{ role: 'user', content: 'x' }])).content).toBe('Answer');
  });

  it('uses a JSON schema for structured output', async () => {
    const f = fakeLlama([{ text: '{"relevant": true}', content: '{"relevant": true}' }]);
    const schema = { type: 'object', properties: { relevant: { type: 'boolean' } } };
    const out = await new OnDeviceModel(cfg, f.init).chatJson([{ role: 'user', content: 'x' }], schema);
    expect(out).toEqual({ relevant: true });
    expect(f.completions[0]).toMatchObject({ response_format: { type: 'json_schema', json_schema: { schema } } });
  });

  it('never runs two completions at once on one context', async () => {
    const f = fakeLlama();
    const m = new OnDeviceModel(cfg, f.init);
    await Promise.all([1, 2, 3].map((i) => m.chat([{ role: 'user', content: String(i) }])));
    expect(f.stats().maxActive).toBe(1);
    expect(f.inits).toHaveLength(1); // loaded once, shared
  });

  it('adds query/document prefixes and truncates long text for embeddings', async () => {
    const f = fakeLlama();
    const m = new OnDeviceModel(cfg, f.init);
    await m.embed(['visa'], 'query');
    await m.embed(['x'.repeat(50_000)]);
    expect(f.embedded[0]).toBe('search_query: visa');
    expect(f.embedded[1]?.startsWith('search_document: ')).toBe(true);
    expect(f.embedded[1]?.length).toBeLessThanOrEqual(2048 * 3);
    expect(f.inits[0]).toMatchObject({ model: '/models/embed.gguf', embedding: true, pooling_type: 'mean' });
  });

  it('retries loading after a failed load', async () => {
    let calls = 0;
    const ok = fakeLlama();
    const init: InitFn = async (p, cb) => {
      calls++;
      if (calls === 1) throw new Error('out of memory');
      return ok.init(p, cb);
    };
    const m = new OnDeviceModel(cfg, init);
    await expect(m.chat([{ role: 'user', content: 'x' }])).rejects.toThrow('out of memory');
    await expect(m.chat([{ role: 'user', content: 'x' }])).resolves.toMatchObject({ content: 'ok' });
  });

  it('release frees both contexts and reloads on next use', async () => {
    const f = fakeLlama();
    const m = new OnDeviceModel(cfg, f.init);
    await m.chat([{ role: 'user', content: 'x' }]);
    await m.embed(['x']);
    await m.release();
    expect(f.stats().released).toBe(2);
    await m.chat([{ role: 'user', content: 'x' }]);
    expect(f.inits).toHaveLength(3);
  });
});

describe('toOaiMessages', () => {
  it('pairs tool results with the tool call ids that produced them', () => {
    const out = toOaiMessages([
      { role: 'user', content: 'q' },
      { role: 'assistant', content: '', toolCalls: [{ name: 'a', arguments: { x: 1 } }, { name: 'b', arguments: {} }] },
      { role: 'tool', toolName: 'a', content: 'ra' },
      { role: 'tool', toolName: 'b', content: 'rb' },
    ]);
    const calls = (out[1] as { tool_calls: { id: string; function: { arguments: string } }[] }).tool_calls;
    expect(calls[0]?.function.arguments).toBe('{"x":1}');
    expect(out[2]).toMatchObject({ role: 'tool', name: 'a', tool_call_id: calls[0]?.id });
    expect(out[3]).toMatchObject({ role: 'tool', name: 'b', tool_call_id: calls[1]?.id });
  });
});

describe('model catalog', () => {
  it('has a default chat and embedding model with direct .gguf links', () => {
    expect(findModel('qwen3-4b-instruct-2507-q4km')?.role).toBe('chat');
    expect(findModel('nomic-embed-text-v1.5-q8')?.role).toBe('embed');
    for (const m of MODELS) {
      expect(m.url).toMatch(/^https:\/\/huggingface\.co\/.+\/resolve\/main\/.+\.gguf$/);
      expect(m.url.endsWith(m.fileName)).toBe(true);
    }
  });
});
