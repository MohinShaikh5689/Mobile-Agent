import {
  customChatModel,
  estimateMemory,
  findModel,
  modelFit,
  recommendedChatModel,
} from '../src/core/llm/models';
import type { InitFn, LlamaLike } from '../src/core/llm/onDevice';
import { OnDeviceModel } from '../src/core/llm/onDevice';

const GB = 1024 ** 3;
// What Device.totalMemory reports, roughly: a bit under the marketed size.
const PHONE_6GB = 5.6 * GB;
const PHONE_8GB = 7.5 * GB; // iPhone 17
const PHONE_12GB = 11.2 * GB; // iPhone 17 Pro

const m = (id: string) => {
  const spec = findModel(id);
  if (!spec) throw new Error(id);
  return spec;
};

describe('model fit', () => {
  it('estimates weights + quantized KV cache + overhead', () => {
    const est = estimateMemory(m('qwen3-4b-instruct-2507-q4km'));
    expect(est / GB).toBeGreaterThan(3.2);
    expect(est / GB).toBeLessThan(4);
  });

  it.each([
    ['qwen3-1.7b-q4km', PHONE_8GB, 'fits'],
    ['qwen3-4b-instruct-2507-q4km', PHONE_8GB, 'fits'],
    ['qwen3-8b-q4km', PHONE_8GB, 'tight'],
    ['qwen3-8b-q4km', PHONE_12GB, 'fits'],
    ['qwen3-8b-q4km', PHONE_6GB, 'too_big'],
    ['qwen3-4b-instruct-2507-q4km', PHONE_6GB, 'tight'],
  ])('%s on %d bytes → %s', (id, ram, fit) => {
    expect(modelFit(m(id), ram)).toBe(fit);
  });

  it('is unknown without RAM info or a size', () => {
    expect(modelFit(m('qwen3-8b-q4km'), null)).toBe('unknown');
    expect(modelFit(customChatModel('', 'https://x.test/a.gguf'), PHONE_8GB)).toBe('unknown');
  });

  it('recommends the largest model that fits comfortably', () => {
    expect(recommendedChatModel(PHONE_12GB).id).toBe('qwen3-8b-q4km');
    expect(recommendedChatModel(PHONE_8GB).id).toBe('qwen3-4b-instruct-2507-q4km');
    expect(recommendedChatModel(PHONE_6GB).id).toBe('qwen3-1.7b-q4km');
    expect(recommendedChatModel(2 * GB).id).toBe('qwen3-1.7b-q4km'); // nothing fits: smallest
    expect(recommendedChatModel(null).id).toBe('qwen3-4b-instruct-2507-q4km');
  });
});

describe('custom models', () => {
  it('builds a chat model from a .gguf link', () => {
    const spec = customChatModel(' My Llama ', 'https://huggingface.co/a/b/resolve/main/Llama-3.1-8B-Q4_K_M.gguf?download=true');
    expect(spec).toMatchObject({ role: 'chat', name: 'My Llama', fileName: 'Llama-3.1-8B-Q4_K_M.gguf', custom: true });
    expect(spec.id).toBe('custom-llama-3.1-8b-q4-k-m.gguf');
  });

  it('rejects links that are not https .gguf files', () => {
    expect(() => customChatModel('', 'http://x.test/a.gguf')).toThrow(/https/);
    expect(() => customChatModel('', 'https://x.test/a.bin')).toThrow(/\.gguf/);
  });

  it('finds custom models next to the built-in ones', () => {
    const spec = customChatModel('', 'https://x.test/a.gguf');
    expect(findModel(spec.id, [spec])).toBe(spec);
    expect(findModel(spec.id)).toBeUndefined();
  });
});

describe('KV cache fallback', () => {
  const ctx: LlamaLike = {
    completion: async () => ({ text: 'ok', content: 'ok' }),
    embedding: async () => ({ embedding: [1] }),
    release: async () => undefined,
  };
  const cfg = { chatModelPath: '/m.gguf', embedModelPath: '/e.gguf', chatCtx: 4096, embedCtx: 2048 };

  it('asks for a q8_0 cache with flash attention', async () => {
    const seen: Record<string, unknown>[] = [];
    const init: InitFn = async (p) => {
      seen.push(p);
      return ctx;
    };
    await new OnDeviceModel({ ...cfg, chatCacheType: 'q8_0' }, init).chat([{ role: 'user', content: 'x' }]);
    expect(seen).toHaveLength(1);
    expect(seen[0]).toMatchObject({ cache_type_k: 'q8_0', cache_type_v: 'q8_0', flash_attn_type: 'on', n_ctx: 4096 });
  });

  it('falls back to an f16 cache when the device rejects q8_0', async () => {
    const seen: Record<string, unknown>[] = [];
    const init: InitFn = async (p) => {
      seen.push(p);
      if (p.cache_type_k === 'q8_0') throw new Error('flash attention not supported');
      return ctx;
    };
    const out = await new OnDeviceModel({ ...cfg, chatCacheType: 'q8_0' }, init).chat([{ role: 'user', content: 'x' }]);
    expect(out.content).toBe('ok');
    expect(seen).toHaveLength(2);
    expect(seen[1]).not.toHaveProperty('cache_type_k');
    expect(seen[1]).toMatchObject({ flash_attn_type: 'auto' });
  });
});
