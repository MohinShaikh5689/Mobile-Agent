export class AuthError extends Error {}
export class GoogleApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

export type TokenProvider = () => Promise<string>;

/** GET/POST to a Google API with the current access token. Retries once on 429/5xx. */
export async function googleFetch<T>(
  getToken: TokenProvider,
  url: string,
  opts: { asText?: boolean; method?: string; body?: unknown } = {},
): Promise<T> {
  for (let attempt = 0; attempt < 2; attempt++) {
    const token = await getToken();
    const res = await fetch(url, {
      method: opts.method ?? 'GET',
      headers: {
        Authorization: `Bearer ${token}`,
        ...(opts.body ? { 'Content-Type': 'application/json' } : {}),
      },
      body: opts.body ? JSON.stringify(opts.body) : undefined,
    });
    if (res.status === 401) throw new AuthError('Google sign-in expired. Sign in again in Settings.');
    if ((res.status === 429 || res.status >= 500) && attempt === 0) {
      await new Promise((r) => setTimeout(r, 1000));
      continue;
    }
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new GoogleApiError(`Google API ${res.status}: ${text.slice(0, 200)}`, res.status);
    }
    return (opts.asText ? await res.text() : await res.json()) as T;
  }
  throw new GoogleApiError('Google API kept failing', 503);
}

export function qs(params: Record<string, string | number | boolean | undefined | string[]>): string {
  const parts: string[] = [];
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === '') continue;
    const values = Array.isArray(v) ? v : [v];
    for (const one of values) parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(String(one))}`);
  }
  return parts.join('&');
}

/** Run async work over items with a small concurrency limit, to stay inside API quotas. */
export async function mapLimit<T, R>(items: T[], limit: number, fn: (t: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i] as T);
    }
  });
  await Promise.all(workers);
  return out;
}
