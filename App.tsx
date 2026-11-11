import { StatusBar } from 'expo-status-bar';
import { useEffect, useMemo, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { buildContainer, kv } from './src/appContainer';
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

  useEffect(() => {
    loadSettings(kv).then(setSettings);
  }, []);

  const container = useMemo(() => (settings ? buildContainer(settings) : null), [settings]);

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
          <ChatScreen agent={container?.agent ?? null} />
        </View>
        {tab === 'settings' && settings && (
          <SettingsScreen settings={settings} onSave={save} container={container} />
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
