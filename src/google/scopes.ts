import type { Source } from '../core/types';

/** The read-only scope each service needs (Safety and approvals: scopes start read-only). */
export const SERVICE_SCOPES: Record<Source, string> = {
  mail: 'https://www.googleapis.com/auth/gmail.readonly',
  docs: 'https://www.googleapis.com/auth/drive.readonly',
  calendar: 'https://www.googleapis.com/auth/calendar.readonly',
};

/**
 * Which services the user actually granted. Google lets people untick individual permissions on
 * its consent screen, so the granted scope string can be a subset of what was asked for.
 * Returns null when Google did not report scopes (then assume what was requested).
 */
export function grantedServices(scope: string | undefined): Record<Source, boolean> | null {
  if (!scope) return null;
  const granted = new Set(scope.split(/\s+/));
  return {
    mail: granted.has(SERVICE_SCOPES.mail),
    docs: granted.has(SERVICE_SCOPES.docs),
    calendar: granted.has(SERVICE_SCOPES.calendar),
  };
}

/** Write scopes, for approved actions only (drafts/sending, creating/changing events). */
export const WRITE_SCOPES = {
  mail: 'https://www.googleapis.com/auth/gmail.compose',
  calendar: 'https://www.googleapis.com/auth/calendar.events',
};

/** Which write permissions were granted; null when Google did not report scopes. */
export function grantedWrites(scope: string | undefined): { mail: boolean; calendar: boolean } | null {
  if (!scope) return null;
  const granted = new Set(scope.split(/\s+/));
  return { mail: granted.has(WRITE_SCOPES.mail), calendar: granted.has(WRITE_SCOPES.calendar) };
}
