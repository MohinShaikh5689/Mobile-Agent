import { activateKeepAwakeAsync, deactivateKeepAwake } from 'expo-keep-awake';
import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Alert, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import type { AppContainer } from '../appContainer';
import { kv, vectorIndex } from '../appContainer';
import { isoDate } from '../core/specialists';
import type { Source } from '../core/types';
import type { Coverage } from '../core/watermark';
import type { ModelSpec } from '../core/llm/models';
import { formatBytes, MODELS } from '../core/llm/models';
import { clientIdsConfigured, useGoogleSignIn } from '../google/auth';
import type { SyncProgress } from '../google/sync';
import { loadCoverage } from '../google/sync';
import type { DownloadHandle } from '../llm/modelStore';
import { deleteModel, downloadModel, isDownloaded } from '../llm/modelStore';
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

/** Download, delete and choose the models that run on the phone. */
function ModelsCard({ settings, onSave, onModelsChanged, t }: {
  settings: Settings;
  onSave: (s: Settings) => void;
  onModelsChanged: () => void;
  t: Theme;
}) {
  const [progress, setProgress] = useState<Record<string, number>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [downloaded, setDownloaded] = useState<Record<string, boolean>>({});
  const [editingUrl, setEditingUrl] = useState<string | null>(null);
  const handles = useRef<Record<string, DownloadHandle>>({});

  const refresh = () => setDownloaded(Object.fromEntries(MODELS.map((m) => [m.id, isDownloaded(m)])));
  useEffect(refresh, []);

  const urlFor = (m: ModelSpec) => settings.modelUrls[m.id] ?? m.url;

  const start = (m: ModelSpec) => {
    setErrors((e) => ({ ...e, [m.id]: '' }));
    setProgress((p) => ({ ...p, [m.id]: 0 }));
    const h = downloadModel(m, urlFor(m), (f) => setProgress((p) => ({ ...p, [m.id]: f })));
    handles.current[m.id] = h;
    h.done
      .then(() => {
        refresh();
        onModelsChanged();
      })
      .catch((e: Error) => setErrors((x) => ({ ...x, [m.id]: e.message })))
      .finally(() => {
        delete handles.current[m.id];
        setProgress((p) => {
          const { [m.id]: _, ...rest } = p;
          return rest;
        });
      });
  };

  const remove = (m: ModelSpec) =>
    Alert.alert(`Delete ${m.name}?`, `Frees ${formatBytes(m.sizeBytes)}. You can download it again later.`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => {
          deleteModel(m);
          refresh();
          onModelsChanged();
        },
      },
    ]);

  const select = (m: ModelSpec) =>
    onSave(m.role === 'chat' ? { ...settings, chatModel: m.id } : { ...settings, embedModel: m.id });

  return (
    <View style={[styles.card, { backgroundColor: t.surface, borderColor: t.border }]}>
      <Text style={[styles.h, { color: t.text }]}>On-device models</Text>
      <Text style={{ color: t.muted }}>
        The agent runs entirely on this phone. Download one chat model and the embedding model, over Wi-Fi.
      </Text>
      {MODELS.map((m) => {
        const selected = m.role === 'chat' ? settings.chatModel === m.id : settings.embedModel === m.id;
        const p = progress[m.id];
        return (
          <View key={m.id} style={[styles.model, { borderColor: selected ? t.accent : t.border }]}>
            <Pressable onPress={() => select(m)} hitSlop={6}>
              <Text style={{ color: t.text, fontWeight: '600' }}>
                {selected ? '● ' : '○ '}
                {m.name}
              </Text>
              <Text style={{ color: t.muted, fontSize: 12 }}>
                {m.role === 'chat' ? 'Chat' : 'Embeddings'} · {formatBytes(m.sizeBytes)} · {m.notes}
              </Text>
            </Pressable>
            <View style={[styles.row, { alignItems: 'center' }]}>
              {p !== undefined ? (
                <>
                  <Text style={{ color: t.muted }}>Downloading {Math.round(p * 100)}%</Text>
                  <Button label="Cancel" onPress={() => handles.current[m.id]?.cancel()} t={t} kind="secondary" />
                </>
              ) : downloaded[m.id] ? (
                <>
                  <Text style={{ color: t.accent }}>Downloaded</Text>
                  <Button label="Delete" onPress={() => remove(m)} t={t} kind="secondary" />
                </>
              ) : (
                <>
                  <Button label="Download" onPress={() => start(m)} t={t} />
                  <Button label="Edit URL" onPress={() => setEditingUrl(editingUrl === m.id ? null : m.id)} t={t} kind="secondary" />
                </>
              )}
            </View>
            {editingUrl === m.id && (
              <Field
                label="Download URL"
                value={urlFor(m)}
                onChange={(v) => onSave({ ...settings, modelUrls: { ...settings.modelUrls, [m.id]: v } })}
                t={t}
                hint="A direct link to the .gguf file."
              />
            )}
            {errors[m.id] ? <Text style={{ color: t.error }}>{errors[m.id]}</Text> : null}
          </View>
        );
      })}
    </View>
  );
}

function coverageText(c: Coverage | undefined): string {
  if (!c) return 'not synced';
  const floor = c.floor !== undefined ? isoDate(c.floor) : 'start';
  const wm = new Date(c.watermark);
  return `${floor} → ${isoDate(c.watermark)} ${wm.toTimeString().slice(0, 5)}`;
}

export function SettingsScreen({ settings, onSave, container, onModelsChanged }: {
  settings: Settings;
  onSave: (s: Settings) => void;
  container: AppContainer | null;
  /** A model was downloaded or deleted: rebuild the agent. */
  onModelsChanged: () => void;
}) {
  const t = useTheme();
  const google = useGoogleSignIn();
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

  const runSync = async () => {
    if (!container) return;
    setSyncing(true);
    // Embedding runs on the phone and takes a while: keep the screen on so the OS doesn't suspend it.
    await activateKeepAwakeAsync('index-sync');
    try {
      await container.indexer.syncAll(setSync);
    } catch (e) {
      setSync({ source: 'mail', phase: 'error', done: 0, total: 0, message: (e as Error).message });
    } finally {
      deactivateKeepAwake('index-sync');
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
      <ModelsCard settings={settings} onSave={onSave} onModelsChanged={onModelsChanged} t={t} />

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
          The first sync can take a while: keep the phone plugged in and the app open.
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
  model: { borderWidth: 1, borderRadius: 10, padding: 12, gap: 8 },
});
