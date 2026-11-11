import { useEffect, useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import type { AppContainer } from '../appContainer';
import { kv, vectorIndex } from '../appContainer';
import { isoDate } from '../core/specialists';
import type { Source } from '../core/types';
import type { Coverage } from '../core/watermark';
import { clientIdsConfigured, useGoogleSignIn } from '../google/auth';
import type { SyncProgress } from '../google/sync';
import { loadCoverage } from '../google/sync';
import type { Settings } from '../storage/settings';
import type { Theme } from './theme';
import { useTheme } from './theme';

function Button({ label, onPress, t, disabled, kind = 'primary' }: {
  label: string;
  onPress: () => void;
  t: Theme;
  disabled?: boolean;
  kind?: 'primary' | 'secondary';
}) {
  const primary = kind === 'primary';
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      style={[
        styles.button,
        primary ? { backgroundColor: t.accent } : { borderColor: t.border, borderWidth: 1 },
        disabled && { opacity: 0.4 },
      ]}
    >
      <Text style={{ color: primary ? t.accentText : t.text, fontWeight: '600' }}>{label}</Text>
    </Pressable>
  );
}

function Field({ label, value, onChange, t, hint }: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  t: Theme;
  hint?: string;
}) {
  return (
    <View style={{ gap: 4 }}>
      <Text style={{ color: t.text, fontWeight: '500' }}>{label}</Text>
      <TextInput
        value={value}
        onChangeText={onChange}
        autoCapitalize="none"
        autoCorrect={false}
        style={[styles.input, { color: t.text, borderColor: t.border, backgroundColor: t.bg }]}
      />
      {hint && <Text style={{ color: t.muted, fontSize: 12 }}>{hint}</Text>}
    </View>
  );
}

function coverageText(c: Coverage | undefined): string {
  if (!c) return 'not synced';
  const floor = c.floor !== undefined ? isoDate(c.floor) : 'start';
  const wm = new Date(c.watermark);
  return `${floor} → ${isoDate(c.watermark)} ${wm.toTimeString().slice(0, 5)}`;
}

export function SettingsScreen({ settings, onSave, container }: {
  settings: Settings;
  onSave: (s: Settings) => void;
  container: AppContainer | null;
}) {
  const t = useTheme();
  const google = useGoogleSignIn();
  const [draft, setDraft] = useState(settings);
  const [modelStatus, setModelStatus] = useState<string | null>(null);
  const [sync, setSync] = useState<SyncProgress | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [coverage, setCoverage] = useState<Partial<Record<Source, Coverage>>>({});
  const [chunks, setChunks] = useState(0);

  const refreshIndexInfo = async () => {
    setCoverage(await loadCoverage(kv));
    setChunks(await vectorIndex.count());
  };

  useEffect(() => {
    refreshIndexInfo();
  }, []);

  useEffect(() => setDraft(settings), [settings]);

  const dirty = JSON.stringify(draft) !== JSON.stringify(settings);

  const testModel = async () => {
    if (!container) return;
    setModelStatus('Checking…');
    try {
      const models = await container.model.listModels();
      const missing = [settings.chatModel, settings.embedModel].filter(
        (m) => !models.some((x) => x === m || x.startsWith(`${m}:`)),
      );
      setModelStatus(
        missing.length
          ? `Connected, but missing: ${missing.join(', ')}. Run "ollama pull <name>" on the laptop.`
          : `Connected. ${models.length} models available.`,
      );
    } catch (e) {
      setModelStatus((e as Error).message);
    }
  };

  const runSync = async () => {
    if (!container) return;
    setSyncing(true);
    try {
      await container.indexer.syncAll(setSync);
    } catch (e) {
      setSync({ source: 'mail', phase: 'error', done: 0, total: 0, message: (e as Error).message });
    } finally {
      setSyncing(false);
      refreshIndexInfo();
    }
  };

  const resetIndex = async () => {
    if (!container) return;
    await container.indexer.reset();
    setSync(null);
    refreshIndexInfo();
  };

  return (
    <ScrollView contentContainerStyle={{ padding: 16, gap: 24 }}>
      <View style={[styles.card, { backgroundColor: t.surface, borderColor: t.border }]}>
        <Text style={[styles.h, { color: t.text }]}>Model server</Text>
        <Text style={{ color: t.muted }}>
          Phase 1 runs the model on your laptop with Ollama. The phone talks to it over Wi-Fi.
        </Text>
        <Field
          label="Ollama URL"
          value={draft.ollamaUrl}
          onChange={(v) => setDraft({ ...draft, ollamaUrl: v })}
          t={t}
          hint="Your laptop's LAN address. Start Ollama with OLLAMA_HOST=0.0.0.0."
        />
        <Field label="Chat model" value={draft.chatModel} onChange={(v) => setDraft({ ...draft, chatModel: v })} t={t} />
        <Field label="Embedding model" value={draft.embedModel} onChange={(v) => setDraft({ ...draft, embedModel: v })} t={t} />
        <View style={styles.row}>
          <Button label="Save" onPress={() => onSave(draft)} t={t} disabled={!dirty} />
          <Button label="Test connection" onPress={testModel} t={t} kind="secondary" disabled={dirty} />
        </View>
        {modelStatus && <Text style={{ color: t.muted }}>{modelStatus}</Text>}
      </View>

      <View style={[styles.card, { backgroundColor: t.surface, borderColor: t.border }]}>
        <Text style={[styles.h, { color: t.text }]}>Google account</Text>
        <Text style={{ color: t.muted }}>Read-only access to Gmail, Drive and Calendar.</Text>
        {!clientIdsConfigured() && (
          <Text style={{ color: t.error }}>
            No OAuth client ID set. Add EXPO_PUBLIC_GOOGLE_*_CLIENT_ID to .env (see README).
          </Text>
        )}
        <View style={styles.row}>
          {google.signedIn ? (
            <Button label="Sign out" onPress={google.signOut} t={t} kind="secondary" />
          ) : (
            <Button label="Sign in with Google" onPress={google.signIn} t={t} disabled={!google.ready} />
          )}
        </View>
        <Text style={{ color: t.muted }}>{google.signedIn ? 'Signed in.' : 'Not signed in.'}</Text>
        {google.error && <Text style={{ color: t.error }}>{google.error}</Text>}
      </View>

      <View style={[styles.card, { backgroundColor: t.surface, borderColor: t.border }]}>
        <Text style={[styles.h, { color: t.text }]}>Local index</Text>
        <Text style={{ color: t.muted }}>
          Older mail and Drive docs are embedded on the phone. Anything newer than the watermark is fetched live.
        </Text>
        <Text style={{ color: t.text }}>Mail: {coverageText(coverage.mail)}</Text>
        <Text style={{ color: t.text }}>Docs: {coverageText(coverage.docs)}</Text>
        <Text style={{ color: t.muted }}>{chunks} chunks stored</Text>
        <View style={styles.row}>
          <Button label={syncing ? 'Syncing…' : 'Sync now'} onPress={runSync} t={t} disabled={syncing || !google.signedIn || !container} />
          <Button label="Reset index" onPress={resetIndex} t={t} kind="secondary" disabled={syncing} />
        </View>
        {sync && (
          <View style={[styles.row, { alignItems: 'center' }]}>
            {syncing && <ActivityIndicator color={t.accent} size="small" />}
            <Text style={{ color: sync.phase === 'error' ? t.error : t.muted, flexShrink: 1 }}>
              {sync.phase === 'error'
                ? `Sync failed: ${sync.message}`
                : `${sync.source}: ${sync.phase}${sync.total ? ` ${sync.done}/${sync.total}` : ''}`}
            </Text>
          </View>
        )}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  card: { borderWidth: 1, borderRadius: 14, padding: 16, gap: 12 },
  h: { fontSize: 17, fontWeight: '600' },
  input: { borderWidth: 1, borderRadius: 8, paddingHorizontal: 10, paddingVertical: 8, fontSize: 15 },
  row: { flexDirection: 'row', gap: 10, flexWrap: 'wrap' },
  button: { borderRadius: 10, paddingHorizontal: 14, paddingVertical: 10 },
});
