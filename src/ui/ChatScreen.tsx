import { useCallback, useEffect, useRef, useState } from 'react';
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
import { Markdown } from './Markdown';
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
          {result.standalone !== result.question && (
            <Text style={[styles.mono, { color: t.muted }]}>understood as: {result.standalone}</Text>
          )}
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

export function ChatScreen({ agent, unavailable, loadStatus }: {
  agent: Agent | null;
  /** Why the agent cannot run yet (models missing), shown instead of the examples. */
  unavailable: string;
  /** Model loading progress, shown while the weights load. */
  loadStatus: string | null;
}) {
  const t = useTheme();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('');
  /** The answer as it streams in. Updates are batched (about 15 a second) to keep scrolling smooth. */
  const [draft, setDraft] = useState('');
  const draftRef = useRef('');
  const draftTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pushDraft = useCallback((text: string) => {
    draftRef.current = text;
    if (draftTimer.current) return;
    draftTimer.current = setTimeout(() => {
      draftTimer.current = null;
      setDraft(draftRef.current);
    }, 66);
  }, []);
  const clearDraft = useCallback(() => {
    if (draftTimer.current) clearTimeout(draftTimer.current);
    draftTimer.current = null;
    draftRef.current = '';
    setDraft('');
  }, []);
  const listRef = useRef<FlatList<Message>>(null);

  // Show the open conversation again after a restart (details of old answers are not kept).
  useEffect(() => {
    if (!agent) return;
    let cancelled = false;
    agent.history().then((turns) => {
      if (cancelled || turns.length === 0) return;
      setMessages((current) =>
        current.length
          ? current
          : turns.map((t, i) => ({ id: `h${t.at}-${i}`, role: t.role === 'user' ? 'user' : 'agent', text: t.text })),
      );
    });
    return () => {
      cancelled = true;
    };
  }, [agent]);

  const newChat = useCallback(async () => {
    if (busy) return;
    await agent?.newConversation();
    setMessages([]);
  }, [agent, busy]);

  const send = useCallback(
    async (text: string) => {
      const q = text.trim();
      if (!q || busy) return;
      if (!agent) {
        // Say why instead of silently ignoring the message.
        setMessages((m) => [
          ...m,
          { id: `${Date.now()}u`, role: 'user', text: q },
          { id: `${Date.now()}e`, role: 'error', text: unavailable || 'The assistant is still loading.' },
        ]);
        setInput('');
        return;
      }
      setInput('');
      setBusy(true);
      setMessages((m) => [...m, { id: `${Date.now()}u`, role: 'user', text: q }]);
      try {
        const result = await agent.ask(q, (e) => {
          if (e.type === 'route' && e.route) setStatus(`${ROUTE_LABEL[e.route.route]}…`);
          if (e.type === 'node' && e.node && NODE_STATUS[e.node]) setStatus(NODE_STATUS[e.node] as string);
          if (e.type === 'step' && e.step?.kind === 'tool') setStatus(`Calling ${e.step.name}…`);
          if (e.type === 'partial') {
            pushDraft(e.text ?? '');
            if (e.text) setStatus('Writing…');
          }
        });
        setMessages((m) => [...m, { id: `${Date.now()}a`, role: 'agent', text: result.answer, result }]);
      } catch (e) {
        setMessages((m) => [...m, { id: `${Date.now()}e`, role: 'error', text: (e as Error).message }]);
      } finally {
        clearDraft();
        setBusy(false);
        setStatus('');
        setTimeout(() => listRef.current?.scrollToEnd({ animated: true }), 50);
      }
    },
    [agent, busy, unavailable, pushDraft, clearDraft],
  );

  return (
    <KeyboardAvoidingView
      style={{ flex: 1 }}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
    >
      {messages.length > 0 && (
        <View style={[styles.chatBar, { borderBottomColor: t.border }]}>
          <Text style={{ color: t.muted, fontSize: 12, flexShrink: 1 }}>
            The assistant remembers this conversation. Start a new chat to change topic.
          </Text>
          <Pressable onPress={newChat} disabled={busy} hitSlop={8} style={{ opacity: busy ? 0.4 : 1 }}>
            <Text style={{ color: t.accent, fontWeight: '600' }}>New chat</Text>
          </Pressable>
        </View>
      )}
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
              {agent ? 'Runs on this phone. Read-only for now. Try one of these:' : unavailable}
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
        ListFooterComponent={
          busy && draft ? (
            <View style={[styles.bubble, { marginTop: 12, backgroundColor: t.surface, borderColor: t.border, borderWidth: 1 }]}>
              <Markdown text={`${draft} ▍`} t={t} />
            </View>
          ) : undefined
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
            {item.role === 'agent' ? (
              <Markdown text={item.text} t={t} />
            ) : (
              <Text style={{ color: item.role === 'error' ? t.error : t.text, fontSize: 15, lineHeight: 21 }} selectable>
                {item.text}
              </Text>
            )}
            {item.result && <Details result={item.result} t={t} />}
          </View>
        )}
      />
      {busy && (
        <View style={styles.status}>
          <ActivityIndicator color={t.accent} size="small" />
          <Text style={{ color: t.muted }}>{loadStatus || status || 'Thinking…'}</Text>
        </View>
      )}
      <View style={[styles.inputRow, { borderTopColor: t.border, backgroundColor: t.surface }]}>
        <TextInput
          value={input}
          onChangeText={setInput}
          placeholder={agent ? 'Ask something…' : 'Ask something (download a model in Settings first)'}
          placeholderTextColor={t.muted}
          style={[styles.input, { color: t.text, backgroundColor: t.bg, borderColor: t.border }]}
          onSubmitEditing={() => send(input)}
          returnKeyType="send"
          // Always editable: the keyboard must open even before a model is ready.
          editable
          multiline
          submitBehavior="submit"
        />
        <Pressable
          onPress={() => send(input)}
          disabled={busy || !input.trim()}
          style={[styles.sendBtn, { backgroundColor: t.accent, opacity: busy || !input.trim() ? 0.4 : 1 }]}
        >
          <Text style={{ color: t.accentText, fontWeight: '600' }}>Send</Text>
        </Pressable>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  bubble: { borderRadius: 14, padding: 12, maxWidth: '100%' },
  chatBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  chip: { borderWidth: 1, borderRadius: 10, paddingVertical: 10, paddingHorizontal: 12 },
  mono: { fontFamily: Platform.select({ ios: 'Menlo', default: 'monospace' }), fontSize: 11 },
  status: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingHorizontal: 16, paddingBottom: 8 },
  inputRow: { flexDirection: 'row', gap: 8, padding: 10, borderTopWidth: StyleSheet.hairlineWidth, alignItems: 'flex-end' },
  input: { flex: 1, borderWidth: 1, borderRadius: 10, paddingHorizontal: 12, paddingVertical: 10, maxHeight: 120, fontSize: 15 },
  sendBtn: { borderRadius: 10, paddingHorizontal: 16, paddingVertical: 12 },
});
