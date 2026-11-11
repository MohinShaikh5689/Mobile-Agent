import type { ConfigContext, ExpoConfig } from 'expo/config';

/**
 * Extends app.json with values that come from the environment at build time.
 *
 * Google sign-in on iOS redirects back to the app on the reversed iOS client ID scheme
 * (com.googleusercontent.apps.<id>), so that scheme must be registered in the app. It is derived
 * from EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID (.env locally, a repository variable in CI).
 */
function reversedClientId(clientId: string | undefined): string | undefined {
  const m = clientId?.trim().match(/^([\w-]+)\.apps\.googleusercontent\.com$/);
  return m ? `com.googleusercontent.apps.${m[1]}` : undefined;
}

export default ({ config }: ConfigContext): ExpoConfig => {
  const base = typeof config.scheme === 'string' ? [config.scheme] : (config.scheme ?? []);
  const googleScheme = reversedClientId(process.env.EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID);
  return {
    ...config,
    name: config.name ?? 'Mobile Agent',
    slug: config.slug ?? 'mobile-agent',
    scheme: googleScheme ? [...base, googleScheme] : base,
  };
};
