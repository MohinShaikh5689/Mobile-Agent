import { activateKeepAwakeAsync, deactivateKeepAwake } from 'expo-keep-awake';
import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Alert, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import type { AppContainer } from '../appContainer';
import { kv, vectorIndex } from '../appContainer';
import { isoDate } from '../core/specialists';
import type { Source } from '../core/types';
import type { Coverage } from '../core/watermark';
import type { Fit, ModelSpec } from '../core/llm/models';
import {
  allModels,
  contextOptions,
  customChatModel,
  effectiveContext,
  estimateMemory,
  formatBytes,
  formatContext,
  modelFit,
  recommendedChatModel,
  withContext,
} from '../core/llm/models';
import { clientIdsConfigured, useGoogleSignIn } from '../google/auth';
import type { SyncProgress } from '../google/sync';
import { loadCoverage } from '../google/sync';
import type { DownloadHandle } from '../llm/modelStore';
import { deviceName, totalRam } from '../llm/device';
import { deleteModel, downloadedSize, downloadModel, isDownloaded } from '../llm/modelStore';
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

const FIT_LABEL: Record<Fit, string> = {
  fits: 'Fits',
  tight: 'Tight: may be closed under memory pressure',
  too_big: 'Too big for this phone',
  unknown: 'Size unknown until downloaded',
};

function fitColor(fit: Fit, t: Theme): string {
  return fit === 'fits' ? t.accent : fit === 'unknown' ? t.muted : t.error;
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
  const [sizes, setSizes] = useState<Record<string, number>>({});
  const [editingUrl, setEditingUrl] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [newName, setNewName] = useState('');
  const [newUrl, setNewUrl] = useState('');
  const [addError, setAddError] = useState<string | null>(null);
  const handles = useRef<Record<string, DownloadHandle>>({});

  const ram = totalRam();
  const recommended = recommendedChatModel(ram);
  const models = allModels(settings.customModels);

  const refresh = () => setSizes(Object.fromEntries(models.map((m) => [m.id, isDownloaded(m) ? downloadedSize(m) : 0])));
  useEffect(refresh, [settings.customModels.length]);

  const urlFor = (m: ModelSpec) => settings.modelUrls[m.id] ?? m.url;
  const sizeOf = (m: ModelSpec) => sizes[m.id] || m.sizeBytes;
  /** The context this model runs with: the user's choice, else the largest that fits. */
  const ctxOf = (m: ModelSpec) => effectiveContext(m, ram, settings.contextTokens[m.id], sizeOf(m));
  const fitOf = (m: ModelSpec, ctx = ctxOf(m)) => modelFit(withContext(m, ctx), ram, sizeOf(m));
  const memOf = (m: ModelSpec, ctx = ctxOf(m)) => estimateMemory(withContext(m, ctx), sizeOf(m));
  const setContext = (m: ModelSpec, n: number) =>
    onSave({ ...settings, contextTokens: { ...settings.contextTokens, [m.id]: n } });

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

  const confirmDownload = (m: ModelSpec) => {
    const fit = fitOf(m);
    if (fit === 'fits' || fit === 'unknown') return start(m);
    Alert.alert(
      fit === 'tight' ? 'This model is tight on this phone' : 'This model is probably too big',
      `${m.name} needs about ${formatBytes(memOf(m))} of memory at a ${formatContext(ctxOf(m))} context, and this phone has ` +
        `${ram ? formatBytes(ram) : 'an unknown amount'}. iOS or Android may close the app while it runs. Download anyway?`,
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Download', onPress: () => start(m) },
      ],
    );
  };

  const remove = (m: ModelSpec) =>
    Alert.alert(`Delete ${m.name}?`, `Frees ${formatBytes(sizes[m.id] || m.sizeBytes)}.`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => {
          deleteModel(m);
          if (m.custom) {
            const customModels = settings.customModels.filter((c) => c.id !== m.id);
            const chatModel = settings.chatModel === m.id ? recommended.id : settings.chatModel;
            onSave({ ...settings, customModels, chatModel });
          }
          refresh();
          onModelsChanged();
        },
      },
    ]);

  const select = (m: ModelSpec) =>
    onSave(m.role === 'chat' ? { ...settings, chatModel: m.id } : { ...settings, embedModel: m.id });

  const addCustom = () => {
    try {
      const spec = customChatModel(newName, newUrl);
      if (models.some((m) => m.id === spec.id)) throw new Error('That model is already in the list.');
      onSave({ ...settings, customModels: [...settings.customModels, spec] });
      setAdding(false);
      setNewName('');
      setNewUrl('');
      setAddError(null);
    } catch (e) {
      setAddError((e as Error).message);
    }
  };

  const groups: { title: string; items: ModelSpec[] }[] = [
    { title: 'Chat models', items: models.filter((m) => m.role === 'chat' && !m.custom) },
    { title: 'Your models', items: models.filter((m) => m.custom) },
    { title: 'Embedding model', items: models.filter((m) => m.role === 'embed') },
  ];

  return (
    <View style={[styles.card, { backgroundColor: t.surface, borderColor: t.border }]}>
      <Text style={[styles.h, { color: t.text }]}>On-device models</Text>
      <Text style={{ color: t.muted }}>
        The agent runs entirely on this phone. Pick one chat model and download it with the embedding model, over Wi-Fi.
        {ram ? ` ${deviceName()}: ${formatBytes(ram)} of RAM.` : ''}
      </Text>
      {groups.map((g) =>
        g.items.length === 0 ? null : (
          <View key={g.title} style={{ gap: 8 }}>
            <Text style={{ color: t.text, fontWeight: '600', marginTop: 4 }}>{g.title}</Text>
            {g.items.map((m) => {
              const selected = m.role === 'chat' ? settings.chatModel === m.id : settings.embedModel === m.id;
              const p = progress[m.id];
              const downloaded = (sizes[m.id] ?? 0) > 0;
              const fit = fitOf(m);
              const size = sizes[m.id] || m.sizeBytes;
              return (
                <View key={m.id} style={[styles.model, { borderColor: selected ? t.accent : t.border }]}>
                  <Pressable onPress={() => select(m)} hitSlop={6} accessibilityRole="radio" accessibilityState={{ selected }}>
                    <Text style={{ color: t.text, fontWeight: '600' }}>
                      {selected ? '● ' : '○ '}
                      {m.name}
                      {m.id === recommended.id ? '  · Recommended' : ''}
                    </Text>
                    <Text style={{ color: t.muted, fontSize: 12 }}>
                      {m.params} · {size ? formatBytes(size) : 'size unknown'}
                      {m.role === 'chat' ? ` · ${formatContext(ctxOf(m))} context` : ''} · {m.notes}
                    </Text>
                    {m.role === 'chat' && (
                      <Text style={{ color: fitColor(fit, t), fontSize: 12 }}>
                        {FIT_LABEL[fit]}
                        {fit !== 'unknown' ? ` (needs about ${formatBytes(memOf(m))})` : ''}
                      </Text>
                    )}
                  </Pressable>
                  {m.role === 'chat' && selected && (
                    <View style={{ gap: 6 }}>
                      <Text style={{ color: t.text, fontSize: 13 }}>
                        Context: how much the model reads at once (conversation, search results, the answer).
                      </Text>
                      <View style={[styles.row, { gap: 6 }]}>
                        {contextOptions(m).map((n) => {
                          const on = ctxOf(m) === n;
                          const f = fitOf(m, n);
                          return (
                            <Pressable
                              key={n}
                              onPress={() => setContext(m, n)}
                              accessibilityRole="radio"
                              accessibilityState={{ selected: on }}
                              style={[
                                styles.chip,
                                { borderColor: on ? t.accent : t.border, backgroundColor: on ? t.userBubble : 'transparent' },
                              ]}
                            >
                              <Text style={{ color: t.text, fontWeight: on ? '700' : '400' }}>{formatContext(n)}</Text>
                              <Text style={{ color: fitColor(f, t), fontSize: 11 }}>
                                {f === 'unknown' ? '—' : formatBytes(memOf(m, n))}
                              </Text>
                            </Pressable>
                          );
                        })}
                      </View>
                      <Text style={{ color: t.muted, fontSize: 12 }}>
                        Longer context uses more memory, and a long prompt takes longer before the answer starts.
                        {settings.contextTokens[m.id] ? '' : ' Chosen automatically: the largest that fits this phone.'}
                      </Text>
                    </View>
                  )}
                  <View style={[styles.row, { alignItems: 'center' }]}>
                    {p !== undefined ? (
                      <>
                        <Text style={{ color: t.muted }}>Downloading {Math.round(p * 100)}%</Text>
                        <Button label="Cancel" onPress={() => handles.current[m.id]?.cancel()} t={t} kind="secondary" />
                      </>
                    ) : downloaded ? (
                      <>
                        <Text style={{ color: t.accent }}>Downloaded</Text>
                        <Button label="Delete" onPress={() => remove(m)} t={t} kind="secondary" />
                      </>
                    ) : (
                      <>
                        <Button label="Download" onPress={() => confirmDownload(m)} t={t} />
                        <Button
                          label="Edit URL"
                          onPress={() => setEditingUrl(editingUrl === m.id ? null : m.id)}
                          t={t}
                          kind="secondary"
                        />
                        {m.custom && <Button label="Remove" onPress={() => remove(m)} t={t} kind="secondary" />}
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
        ),
      )}
      {adding ? (
        <View style={[styles.model, { borderColor: t.border }]}>
          <Field label="Name" value={newName} onChange={setNewName} t={t} hint="Optional, for your list." />
          <Field
            label="GGUF link"
            value={newUrl}
            onChange={setNewUrl}
            t={t}
            hint="A direct https link to a .gguf chat model, e.g. a Hugging Face …/resolve/main/…Q4_K_M.gguf link. Q4_K_M is a good size/quality balance."
          />
          {addError && <Text style={{ color: t.error }}>{addError}</Text>}
          <View style={styles.row}>
            <Button label="Add" onPress={addCustom} t={t} disabled={!newUrl.trim()} />
            <Button label="Cancel" onPress={() => setAdding(false)} t={t} kind="secondary" />
          </View>
        </View>
      ) : (
        <Button label="Add a model by link" onPress={() => setAdding(true)} t={t} kind="secondary" />
      )}
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
        {google.signedIn && google.granted && (
          <View style={{ gap: 4 }}>
            {(
              [
                ['mail', 'Gmail'],
                ['docs', 'Google Drive'],
                ['calendar', 'Google Calendar'],
              ] as const
            ).map(([key, label]) => (
              <Text key={key} style={{ color: google.granted?.[key] ? t.accent : t.error }}>
                {google.granted?.[key] ? '✓' : '✗'} {label}
              </Text>
            ))}
            {!(google.granted.mail && google.granted.docs && google.granted.calendar) && (
              <Text style={{ color: t.error, fontSize: 12 }}>
                Some access was not granted. Sign out, sign in again, and tick every box on Google's permission screen.
              </Text>
            )}
          </View>
        )}
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
  chip: { borderWidth: 1, borderRadius: 8, paddingHorizontal: 10, paddingVertical: 6, alignItems: 'center', minWidth: 64 },
});
