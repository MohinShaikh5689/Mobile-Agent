import type { ConfigContext, ExpoConfig } from 'expo/config';

/**
 * Extends app.json with values that come from the environment at build time.
 *
 * Google sign-in redirects back to the app on a custom URL scheme, which must be registered:
 * - iOS: the reversed iOS client ID (com.googleusercontent.apps.<id>), derived from
 *   EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID (.env locally, a repository variable in CI).
 * - Android: the package name (com.mobileagent.app:/oauthredirect).
 */
function reversedClientId(clientId: string | undefined): string | undefined {
  const m = clientId?.trim().match(/^([\w-]+)\.apps\.googleusercontent\.com$/);
  return m ? `com.googleusercontent.apps.${m[1]}` : undefined;
}

export default ({ config }: ConfigContext): ExpoConfig => {
  const base = typeof config.scheme === 'string' ? [config.scheme] : (config.scheme ?? []);
  const extra = [config.android?.package, reversedClientId(process.env.EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID)];
  const scheme = [...base];
  for (const s of extra) if (s && !scheme.includes(s)) scheme.push(s);
  return {
    ...config,
    name: config.name ?? 'Mobile Agent',
    slug: config.slug ?? 'mobile-agent',
    scheme,
    plugins: [...(config.plugins ?? []), './plugins/withReleaseSigning'],
  };
};
