import { useState } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import type { ProposedAction } from '../core/actions';
import { ACTION_LABEL, applyEdits, describeAction, editableFields } from '../core/actions';
import { Markdown } from './Markdown';
import type { Theme } from './theme';

const FIELD_LABELS: Record<string, string> = {
  to: 'To',
  cc: 'Cc',
  subject: 'Subject',
  body: 'Message',
  title: 'Title',
  date: 'Date (YYYY-MM-DD)',
  start: 'Start (HH:MM, empty for all day)',
  end: 'End (HH:MM)',
  attendees: 'Invite (emails, comma-separated)',
  location: 'Location',
};

/**
 * The approval gate's UI: shows exactly what will happen, lets the user edit it, and runs it only
 * when they tap the action button. Nothing is written before that.
 */
export function ActionCard({ action, t, onApprove, onCancel }: {
  action: ProposedAction;
  t: Theme;
  onApprove: (action: ProposedAction) => Promise<void>;
  onCancel: () => Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const [fields, setFields] = useState(() => editableFields(action));
  const [current, setCurrent] = useState(action);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const saveEdits = () => {
    const res = applyEdits(current, fields);
    if (!res.ok) return setError(res.error);
    setCurrent(res.action);
    setError(null);
    setEditing(false);
  };

  const run = async (fn: () => Promise<void>) => {
    setBusy(true);
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const destructive = current.kind === 'send_email';

  return (
    <View style={[styles.card, { borderColor: destructive ? t.error : t.accent, backgroundColor: t.bg }]}>
      <Text style={{ color: t.muted, fontSize: 12, fontWeight: '600' }}>NEEDS YOUR APPROVAL</Text>
      {editing ? (
        <View style={{ gap: 8 }}>
          {Object.entries(fields).map(([key, value]) => (
            <View key={key} style={{ gap: 2 }}>
              <Text style={{ color: t.muted, fontSize: 12 }}>{FIELD_LABELS[key] ?? key}</Text>
              <TextInput
                value={value}
                onChangeText={(v) => setFields((f) => ({ ...f, [key]: v }))}
                multiline={key === 'body'}
                autoCapitalize={key === 'body' || key === 'title' || key === 'subject' ? 'sentences' : 'none'}
                autoCorrect={key === 'body' || key === 'subject' || key === 'title'}
                style={[
                  styles.input,
                  { color: t.text, borderColor: t.border, backgroundColor: t.surface },
                  key === 'body' && { minHeight: 100, textAlignVertical: 'top' },
                ]}
              />
            </View>
          ))}
        </View>
      ) : (
        <Markdown text={describeAction(current)} t={t} />
      )}
      {error && <Text style={{ color: t.error }}>{error}</Text>}
      <View style={styles.row}>
        {busy ? (
          <ActivityIndicator color={t.accent} />
        ) : editing ? (
          <>
            <Pressable onPress={saveEdits} style={[styles.btn, { backgroundColor: t.accent }]}>
              <Text style={{ color: t.accentText, fontWeight: '600' }}>Done</Text>
            </Pressable>
            <Pressable onPress={() => setEditing(false)} style={[styles.btn, { borderColor: t.border, borderWidth: 1 }]}>
              <Text style={{ color: t.text }}>Discard edits</Text>
            </Pressable>
          </>
        ) : (
          <>
            <Pressable
              onPress={() => run(() => onApprove(current))}
              style={[styles.btn, { backgroundColor: destructive ? t.error : t.accent }]}
              accessibilityHint="Runs this action now"
            >
              <Text style={{ color: t.accentText, fontWeight: '600' }}>{ACTION_LABEL[current.kind]}</Text>
            </Pressable>
            <Pressable
              onPress={() => {
                setFields(editableFields(current));
                setEditing(true);
              }}
              style={[styles.btn, { borderColor: t.border, borderWidth: 1 }]}
            >
              <Text style={{ color: t.text }}>Edit</Text>
            </Pressable>
            <Pressable onPress={() => run(onCancel)} style={[styles.btn, { borderColor: t.border, borderWidth: 1 }]}>
              <Text style={{ color: t.text }}>Cancel</Text>
            </Pressable>
          </>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: { marginTop: 10, borderWidth: 1, borderRadius: 12, padding: 12, gap: 10 },
  row: { flexDirection: 'row', gap: 8, flexWrap: 'wrap', alignItems: 'center' },
  btn: { borderRadius: 10, paddingHorizontal: 14, paddingVertical: 10 },
  input: { borderWidth: 1, borderRadius: 8, paddingHorizontal: 10, paddingVertical: 8, fontSize: 15 },
});
