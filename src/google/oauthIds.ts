/**
 * Google iOS OAuth clients accept a redirect on the "reversed client ID" scheme:
 *   123-abc.apps.googleusercontent.com  →  com.googleusercontent.apps.123-abc:/oauthredirect
 * Unlike the bundle-ID redirect, it still works after a sideloading tool rewrites the bundle ID.
 * app.config.ts registers the same scheme (it repeats this function, since the config loader
 * runs outside the app bundle).
 */
export function reversedClientId(clientId: string | undefined): string | undefined {
  const m = clientId?.trim().match(/^([\w-]+)\.apps\.googleusercontent\.com$/);
  return m ? `com.googleusercontent.apps.${m[1]}` : undefined;
}

export function iosRedirectUri(iosClientId: string | undefined): string | undefined {
  const scheme = reversedClientId(iosClientId);
  return scheme ? `${scheme}:/oauthredirect` : undefined;
}
