/**
 * Errors the agent cannot work around: retrying or rephrasing will not help, so the tool loop
 * stops and shows the message to the user as is (instead of the model paraphrasing it as
 * "I don't have access").
 */
export class AuthError extends Error {
  readonly fatal = true;
}

export class GoogleApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

/** A permission problem with a clear fix: the API is off in the Cloud project, or the scope was not granted. */
export class GoogleAccessError extends GoogleApiError {
  readonly fatal = true;
}

const API_NAMES: Record<string, string> = {
  'gmail.googleapis.com': 'Gmail',
  'www.googleapis.com/drive': 'Google Drive',
  'www.googleapis.com/calendar': 'Google Calendar',
};

function apiName(url: string): string {
  for (const [k, v] of Object.entries(API_NAMES)) if (url.includes(k)) return v;
  return 'Google';
}

/** Reads Google's JSON error body and turns the common 403s into a message that says what to do. */
export function explainGoogleError(status: number, body: string, url: string): GoogleApiError {
  const name = apiName(url);
  let reason = '';
  let message = body.slice(0, 200);
  try {
    const e = (JSON.parse(body) as { error?: { message?: string; status?: string; errors?: { reason?: string }[]; details?: { reason?: string }[] } }).error;
    reason = [e?.status, ...(e?.errors ?? []).map((x) => x.reason), ...(e?.details ?? []).map((x) => x.reason)]
      .filter(Boolean)
      .join(' ');
    message = e?.message ?? message;
  } catch {
    // not JSON
  }
  if (status === 403 && /SERVICE_DISABLED|accessNotConfigured|has not been used|is disabled/i.test(`${reason} ${message}`)) {
    return new GoogleAccessError(
      `The ${name} API is not enabled in your Google Cloud project. Enable it under APIs & Services → Library, wait a minute, then try again.`,
      status,
    );
  }
  if (status === 403 && /ACCESS_TOKEN_SCOPE_INSUFFICIENT|insufficientPermissions|insufficient authentication scopes/i.test(`${reason} ${message}`)) {
    return new GoogleAccessError(
      `${name} access was not granted when you signed in. In Settings, sign out, sign in again, and tick the ${name} box on Google's permission screen.`,
      status,
    );
  }
  return new GoogleApiError(`${name} API error ${status}: ${message}`, status);
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
      throw explainGoogleError(res.status, text, url);
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
