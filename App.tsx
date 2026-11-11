import { StatusBar } from 'expo-status-bar';
import { useCallback, useEffect, useState } from 'react';
import { AppState, Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import type { AppContainer } from './src/appContainer';
import { buildContainer, kv } from './src/appContainer';
import { onModelLoadProgress, releaseModelMemory } from './src/llm/runtime';
import type { Settings } from './src/storage/settings';
import { loadSettings, saveSettings } from './src/storage/settings';
import { ChatScreen } from './src/ui/ChatScreen';
import { SettingsScreen } from './src/ui/SettingsScreen';
import { useTheme } from './src/ui/theme';

type Tab = 'chat' | 'settings';

export default function App() {
  const t = useTheme();
  const [tab, setTab] = useState<Tab>('chat');
  const [settings, setSettings] = useState<Settings | null>(null);
  const [container, setContainer] = useState<AppContainer | null>(null);
  const [unavailable, setUnavailable] = useState<string>('Loading…');
  const [loadStatus, setLoadStatus] = useState<string | null>(null);

  useEffect(() => {
    loadSettings(kv).then(setSettings);
  }, []);

  /** (Re)builds the agent; called when settings change or a model finishes downloading. */
  const rebuild = useCallback(async (s: Settings) => {
    const res = await buildContainer(s);
    if (res.ok) {
      setContainer(res.container);
      setUnavailable('');
    } else {
      setContainer(null);
      setUnavailable(res.reason);
    }
  }, []);

  useEffect(() => {
    if (settings) rebuild(settings);
  }, [settings, rebuild]);

  useEffect(() => {
    onModelLoadProgress((which, p) => {
      setLoadStatus(p >= 100 ? null : `Loading ${which === 'chat' ? 'chat' : 'embedding'} model ${Math.round(p)}%`);
    });
    // iOS ends memory-heavy apps in the background first: let go of the model weights there.
    const sub = AppState.addEventListener('change', (state) => {
      if (state === 'background') releaseModelMemory();
    });
    return () => {
      onModelLoadProgress(null);
      sub.remove();
    };
  }, []);

  const save = async (s: Settings) => {
    await saveSettings(kv, s);
    setSettings(s);
  };

  return (
    <SafeAreaProvider>
      <SafeAreaView style={{ flex: 1, backgroundColor: t.bg }} edges={['top', 'bottom']}>
        <StatusBar style="auto" />
        <View style={[styles.header, { borderBottomColor: t.border }]}>
          <Text style={[styles.title, { color: t.text }]}>Mobile Agent</Text>
          <View style={styles.tabs}>
            {(['chat', 'settings'] as Tab[]).map((name) => (
              <Pressable
                key={name}
                onPress={() => setTab(name)}
                style={[styles.tab, tab === name && { backgroundColor: t.userBubble }]}
              >
                <Text style={{ color: tab === name ? t.text : t.muted, fontWeight: '500' }}>
                  {name === 'chat' ? 'Chat' : 'Settings'}
                </Text>
              </Pressable>
            ))}
          </View>
        </View>
        <View style={{ flex: 1, display: tab === 'chat' ? 'flex' : 'none' }}>
          <ChatScreen agent={container?.agent ?? null} unavailable={unavailable} loadStatus={loadStatus} />
        </View>
        {tab === 'settings' && settings && (
          <SettingsScreen
            settings={settings}
            onSave={save}
            container={container}
            onModelsChanged={() => rebuild(settings)}
          />
        )}
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  title: { fontSize: 18, fontWeight: '700' },
  tabs: { flexDirection: 'row', gap: 4 },
  tab: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 8 },
});
