import { useCallback, useRef, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  KeyboardAvoidingView,
  Linking,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import type { TraceStep } from '../core/agent';
import type { AgentAnswer } from '../core/orchestrator';
import type { RetrievalTraceEntry } from '../core/retriever';
import type { Route } from '../core/router';
import { isoDate } from '../core/specialists';
import type { Agent } from '../core/orchestrator';
import type { Theme } from './theme';
import { useTheme } from './theme';

interface Message {
  id: string;
  role: 'user' | 'agent' | 'error';
  text: string;
  result?: AgentAnswer;
}

const ROUTE_LABEL: Record<Route, string> = {
  structured: 'Exact lookup',
  fuzzy: 'Recall (index + live)',
  multi_step: 'Multi-step',
  action: 'Action',
};

/** What the user sees while each graph node's successor runs. */
const NODE_STATUS: Record<string, string> = {
  retrieve: 'Checking the results…',
  grade: 'Writing the answer…',
  rewrite: 'Searching again with a better query…',
  planner: 'Working through the plan…',
  run_step: 'Next step…',
};

const EXAMPLES = [
  "What's on my calendar tomorrow?",
  'Any unread mail from today?',
  'Find the thread about the visa paperwork',
  'What did the vendor say about pricing?',
];

function windowText(w: { start?: number; end?: number } | null): string {
  if (!w) return 'all time';
  const s = w.start !== undefined ? isoDate(w.start) : '…';
  const e = w.end !== undefined ? isoDate(w.end) : 'now';
  return `${s} → ${e}`;
}

function retrievalLine(r: RetrievalTraceEntry): string {
  const parts = [`${r.source}:`];
  if (r.indexWindow) parts.push(`index ${windowText(r.indexWindow)} (${r.indexHits})`);
  if (r.liveWindows.length) parts.push(`live ${r.liveWindows.map(windowText).join(' + ')} (${r.liveHits})`);
  if (!r.coverage) parts.push('· not synced yet');
  if (r.fallback) parts.push('· weak index result, checked live');
  if (r.error) parts.push(`· live error: ${r.error}`);
  return parts.join(' ');
}

function stepLine(s: TraceStep): string | null {
  if (s.kind === 'tool') return `${s.ok ? '✓' : '✗'} ${s.name}(${JSON.stringify(s.args)})`;
  if (s.kind === 'note') return `• ${s.message}`;
  return null;
}

function Details({ result, t }: { result: AgentAnswer; t: Theme }) {
  const [open, setOpen] = useState(false);
  const steps = result.steps.map(stepLine).filter((x): x is string => x !== null);
  return (
    <View style={{ marginTop: 8 }}>
      <Pressable onPress={() => setOpen((o) => !o)} hitSlop={8}>
        <Text style={{ color: t.muted, fontSize: 12 }}>
          {ROUTE_LABEL[result.route.route]} · {result.route.services.join(', ') || '—'} {open ? '▾' : '▸'}
        </Text>
      </Pressable>
      {open && (
        <View style={{ marginTop: 6, gap: 3 }}>
          <Text style={[styles.mono, { color: t.muted }]}>why: {result.route.reason}</Text>
          <Text style={[styles.mono, { color: t.muted }]}>graph: {result.path.join(' → ')}</Text>
          {result.plan.length > 0 && (
            <Text style={[styles.mono, { color: t.muted }]}>
              plan: {result.plan.map((p) => `${p.service}: ${p.task}`).join(' | ')}
            </Text>
          )}
          {result.route.window && (
            <Text style={[styles.mono, { color: t.muted }]}>window: {windowText(result.route.window)}</Text>
          )}
          {result.retrieval.map((r) => (
            <Text key={r.source} style={[styles.mono, { color: t.muted }]}>
              {retrievalLine(r)}
            </Text>
          ))}
          {steps.map((s, i) => (
            <Text key={i} style={[styles.mono, { color: t.muted }]} numberOfLines={2}>
              {s}
            </Text>
          ))}
          {result.sources.map((s, i) => (
            <Pressable key={`${s.source}:${s.id}`} onPress={() => s.url && Linking.openURL(s.url)}>
              <Text style={[styles.mono, { color: t.accent }]} numberOfLines={1}>
                [{i + 1}] {s.title} · {s.origin} · {s.score.toFixed(2)}
              </Text>
            </Pressable>
          ))}
        </View>
      )}
    </View>
  );
}

export function ChatScreen({ agent }: { agent: Agent | null }) {
  const t = useTheme();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('');
  const listRef = useRef<FlatList<Message>>(null);

  const send = useCallback(
    async (text: string) => {
      const q = text.trim();
      if (!q || busy || !agent) return;
      setInput('');
      setBusy(true);
      setMessages((m) => [...m, { id: `${Date.now()}u`, role: 'user', text: q }]);
      try {
        const result = await agent.ask(q, (e) => {
          if (e.type === 'route' && e.route) setStatus(`${ROUTE_LABEL[e.route.route]}…`);
          if (e.type === 'node' && e.node && NODE_STATUS[e.node]) setStatus(NODE_STATUS[e.node] as string);
          if (e.type === 'step' && e.step?.kind === 'tool') setStatus(`Calling ${e.step.name}…`);
        });
        setMessages((m) => [...m, { id: `${Date.now()}a`, role: 'agent', text: result.answer, result }]);
      } catch (e) {
        setMessages((m) => [...m, { id: `${Date.now()}e`, role: 'error', text: (e as Error).message }]);
      } finally {
        setBusy(false);
        setStatus('');
        setTimeout(() => listRef.current?.scrollToEnd({ animated: true }), 50);
      }
    },
    [agent, busy],
  );

  return (
    <KeyboardAvoidingView
      style={{ flex: 1 }}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
    >
      <FlatList
        ref={listRef}
        data={messages}
        keyExtractor={(m) => m.id}
        contentContainerStyle={{ padding: 16, gap: 12, flexGrow: 1 }}
        onContentSizeChange={() => listRef.current?.scrollToEnd({ animated: true })}
        ListEmptyComponent={
          <View style={{ flex: 1, justifyContent: 'center', gap: 10 }}>
            <Text style={{ color: t.text, fontSize: 20, fontWeight: '600' }}>Ask about your mail, docs and calendar</Text>
            <Text style={{ color: t.muted }}>
              {agent ? 'Read-only for now. Try one of these:' : 'Set up the model server and Google sign-in in Settings first.'}
            </Text>
            {agent &&
              EXAMPLES.map((ex) => (
                <Pressable
                  key={ex}
                  onPress={() => send(ex)}
                  style={[styles.chip, { borderColor: t.border, backgroundColor: t.surface }]}
                >
                  <Text style={{ color: t.text }}>{ex}</Text>
                </Pressable>
              ))}
          </View>
        }
        renderItem={({ item }) => (
          <View
            style={[
              styles.bubble,
              item.role === 'user'
                ? { alignSelf: 'flex-end', backgroundColor: t.userBubble }
                : { alignSelf: 'stretch', backgroundColor: t.surface, borderColor: t.border, borderWidth: 1 },
            ]}
          >
            <Text style={{ color: item.role === 'error' ? t.error : t.text, fontSize: 15, lineHeight: 21 }} selectable>
              {item.text}
            </Text>
            {item.result && <Details result={item.result} t={t} />}
          </View>
        )}
      />
      {busy && (
        <View style={styles.status}>
          <ActivityIndicator color={t.accent} size="small" />
          <Text style={{ color: t.muted }}>{status || 'Thinking…'}</Text>
        </View>
      )}
      <View style={[styles.inputRow, { borderTopColor: t.border, backgroundColor: t.surface }]}>
        <TextInput
          value={input}
          onChangeText={setInput}
          placeholder="Ask something…"
          placeholderTextColor={t.muted}
          style={[styles.input, { color: t.text, backgroundColor: t.bg, borderColor: t.border }]}
          onSubmitEditing={() => send(input)}
          returnKeyType="send"
          editable={!busy && Boolean(agent)}
          multiline
          submitBehavior="submit"
        />
        <Pressable
          onPress={() => send(input)}
          disabled={busy || !input.trim() || !agent}
          style={[styles.sendBtn, { backgroundColor: t.accent, opacity: busy || !input.trim() || !agent ? 0.4 : 1 }]}
        >
          <Text style={{ color: t.accentText, fontWeight: '600' }}>Send</Text>
        </Pressable>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  bubble: { borderRadius: 14, padding: 12, maxWidth: '100%' },
  chip: { borderWidth: 1, borderRadius: 10, paddingVertical: 10, paddingHorizontal: 12 },
  mono: { fontFamily: Platform.select({ ios: 'Menlo', default: 'monospace' }), fontSize: 11 },
  status: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingHorizontal: 16, paddingBottom: 8 },
  inputRow: { flexDirection: 'row', gap: 8, padding: 10, borderTopWidth: StyleSheet.hairlineWidth, alignItems: 'flex-end' },
  input: { flex: 1, borderWidth: 1, borderRadius: 10, paddingHorizontal: 12, paddingVertical: 10, maxHeight: 120, fontSize: 15 },
  sendBtn: { borderRadius: 10, paddingHorizontal: 16, paddingVertical: 12 },
});
