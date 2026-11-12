import * as AuthSession from 'expo-auth-session';
import * as Google from 'expo-auth-session/providers/google';
import * as SecureStore from 'expo-secure-store';
import * as WebBrowser from 'expo-web-browser';
import { useCallback, useEffect, useState } from 'react';
import { Platform } from 'react-native';
import { AuthError } from './http';
import type { Source } from '../core/types';
import { iosRedirectUri } from './oauthIds';
import { grantedServices, grantedWrites, SERVICE_SCOPES, WRITE_SCOPES } from './scopes';

WebBrowser.maybeCompleteAuthSession();

/** OAuth scopes start read-only and widen only when write actions are built (Safety and approvals). */
/**
 * Read scopes for lookups, plus the two write scopes used only after the user approves an action
 * (Safety and approvals: scopes widened when write actions were built).
 */
export const SCOPES = [
  'openid',
  'email',
  SERVICE_SCOPES.mail,
  SERVICE_SCOPES.docs,
  SERVICE_SCOPES.calendar,
  WRITE_SCOPES.mail,
  WRITE_SCOPES.calendar,
];

const KEY = 'google.tokens';

interface StoredTokens {
  accessToken: string;
  refreshToken?: string;
  /** Epoch ms. */
  expiresAt: number;
  /** Space-separated scopes Google granted (the user can untick some on the consent screen). */
  scope?: string;
}

const CLIENT_IDS = {
  ios: process.env.EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID,
  android: process.env.EXPO_PUBLIC_GOOGLE_ANDROID_CLIENT_ID,
  web: process.env.EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID,
};

function platformClientId(): string | undefined {
  return Platform.select({ ios: CLIENT_IDS.ios, android: CLIENT_IDS.android, default: CLIENT_IDS.web });
}

export function clientIdsConfigured(): boolean {
  return Boolean(platformClientId());
}

async function load(): Promise<StoredTokens | null> {
  const raw = await SecureStore.getItemAsync(KEY);
  return raw ? (JSON.parse(raw) as StoredTokens) : null;
}

async function save(t: StoredTokens): Promise<void> {
  await SecureStore.setItemAsync(KEY, JSON.stringify(t));
}

export async function signOut(): Promise<void> {
  await SecureStore.deleteItemAsync(KEY);
}

let refreshing: Promise<string> | null = null;

/** Returns a valid access token, refreshing it when it is about to expire. */
export async function getAccessToken(): Promise<string> {
  const t = await load();
  if (!t) throw new AuthError('Not signed in to Google. Sign in under Settings.');
  if (t.expiresAt - 60_000 > Date.now()) return t.accessToken;
  const clientId = platformClientId();
  if (!t.refreshToken || !clientId) throw new AuthError('Google sign-in expired. Sign in again in Settings.');

  refreshing ??= (async () => {
    try {
      const res = await AuthSession.refreshAsync(
        { clientId, refreshToken: t.refreshToken },
        Google.discovery,
      );
      await save({
        accessToken: res.accessToken,
        refreshToken: res.refreshToken ?? t.refreshToken,
        expiresAt: Date.now() + (res.expiresIn ?? 3600) * 1000,
        scope: res.scope ?? t.scope,
      });
      return res.accessToken;
    } catch {
      throw new AuthError('Google sign-in expired. Sign in again in Settings.');
    } finally {
      refreshing = null;
    }
  })();
  return refreshing;
}

/** Sign-in hook for the Settings screen. */
export function useGoogleSignIn() {
  const [signedIn, setSignedIn] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [request, response, promptAsync] = Google.useAuthRequest({
    iosClientId: CLIENT_IDS.ios,
    androidClientId: CLIENT_IDS.android,
    webClientId: CLIENT_IDS.web,
    // iOS: redirect on the reversed client ID scheme (registered by app.config.ts), which keeps
    // working when a sideloading tool rewrites the bundle ID. Android keeps the package-name default.
    ...(Platform.OS === 'ios' && iosRedirectUri(CLIENT_IDS.ios) ? { redirectUri: iosRedirectUri(CLIENT_IDS.ios) } : {}),
    scopes: SCOPES,
    // Ask for a refresh token so the user is not signed out every hour.
    extraParams: { access_type: 'offline', prompt: 'consent' },
  });

  const [granted, setGranted] = useState<Record<Source, boolean> | null>(null);
  const [writes, setWrites] = useState<{ mail: boolean; calendar: boolean } | null>(null);

  useEffect(() => {
    load().then((t) => {
      setSignedIn(Boolean(t));
      setGranted(grantedServices(t?.scope));
      setWrites(grantedWrites(t?.scope));
    });
  }, []);

  useEffect(() => {
    if (response?.type === 'success' && response.authentication) {
      const a = response.authentication;
      save({
        accessToken: a.accessToken,
        refreshToken: a.refreshToken,
        expiresAt: Date.now() + (a.expiresIn ?? 3600) * 1000,
        scope: a.scope,
      }).then(() => {
        setSignedIn(true);
        setGranted(grantedServices(a.scope));
        setWrites(grantedWrites(a.scope));
        setError(null);
      });
    } else if (response?.type === 'error') {
      setError(response.error?.message ?? 'Sign-in failed');
    }
  }, [response]);

  const signIn = useCallback(() => promptAsync(), [promptAsync]);
  const doSignOut = useCallback(async () => {
    await signOut();
    setSignedIn(false);
    setGranted(null);
    setWrites(null);
  }, []);

  return { ready: Boolean(request) && clientIdsConfigured(), signedIn, granted, writes, error, signIn, signOut: doSignOut };
}
