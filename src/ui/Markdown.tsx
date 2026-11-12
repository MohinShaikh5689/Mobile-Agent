import { memo, useMemo } from 'react';
import { Linking, Platform, StyleSheet, Text, View } from 'react-native';
import type { Block, Inline } from '../core/markdown';
import { parseMarkdown } from '../core/markdown';
import type { Theme } from './theme';

const MONO = Platform.select({ ios: 'Menlo', default: 'monospace' });

function InlineNodes({ nodes, t }: { nodes: Inline[]; t: Theme }) {
  return (
    <>
      {nodes.map((n, i) => {
        switch (n.type) {
          case 'text':
            return n.text;
          case 'bold':
            return (
              <Text key={i} style={{ fontWeight: '700' }}>
                <InlineNodes nodes={n.children} t={t} />
              </Text>
            );
          case 'italic':
            return (
              <Text key={i} style={{ fontStyle: 'italic' }}>
                <InlineNodes nodes={n.children} t={t} />
              </Text>
            );
          case 'code':
            return (
              <Text key={i} style={[styles.inlineCode, { backgroundColor: t.bg, color: t.text }]}>
                {n.text}
              </Text>
            );
          case 'link':
            return (
              <Text
                key={i}
                style={{ color: t.accent, textDecorationLine: 'underline' }}
                onPress={() => Linking.openURL(n.href)}
                accessibilityRole="link"
              >
                <InlineNodes nodes={n.children} t={t} />
              </Text>
            );
        }
      })}
    </>
  );
}

function BlockView({ block, t, first }: { block: Block; t: Theme; first: boolean }) {
  const gap = first ? null : { marginTop: 8 };
  const body = { color: t.text, fontSize: 15, lineHeight: 21 };
  switch (block.type) {
    case 'heading':
      return (
        <Text style={[body, gap, { fontWeight: '700', fontSize: block.level === 1 ? 18 : block.level === 2 ? 17 : 16 }]} selectable>
          <InlineNodes nodes={block.content} t={t} />
        </Text>
      );
    case 'paragraph':
      return (
        <Text style={[body, gap]} selectable>
          <InlineNodes nodes={block.content} t={t} />
        </Text>
      );
    case 'quote':
      return (
        <View style={[gap, styles.quote, { borderLeftColor: t.border }]}>
          <Text style={[body, { color: t.muted }]} selectable>
            <InlineNodes nodes={block.content} t={t} />
          </Text>
        </View>
      );
    case 'code':
      return (
        <View style={[gap, styles.codeBlock, { backgroundColor: t.bg, borderColor: t.border }]}>
          <Text style={{ fontFamily: MONO, fontSize: 13, color: t.text }} selectable>
            {block.text}
          </Text>
        </View>
      );
    case 'rule':
      return <View style={[gap, { height: StyleSheet.hairlineWidth, backgroundColor: t.border }]} />;
    case 'list': {
      let counter = 0;
      return (
        <View style={[gap, { gap: 4 }]}>
          {block.items.map((item, i) => {
            if (item.depth === 0) counter = item.number ?? counter + 1;
            const marker = block.ordered && item.depth === 0 ? `${counter}.` : item.depth === 0 ? '•' : '◦';
            return (
              <View key={i} style={{ flexDirection: 'row', paddingLeft: item.depth * 18 }}>
                <Text style={[body, { width: block.ordered ? 24 : 16 }]}>{marker}</Text>
                <Text style={[body, { flex: 1 }]} selectable>
                  <InlineNodes nodes={item.content} t={t} />
                </Text>
              </View>
            );
          })}
        </View>
      );
    }
  }
}

/** Renders a chat answer written in Markdown. Safe to call with half-streamed text. */
export const Markdown = memo(function Markdown({ text, t }: { text: string; t: Theme }) {
  const blocks = useMemo(() => parseMarkdown(text), [text]);
  return (
    <View>
      {blocks.map((b, i) => (
        <BlockView key={i} block={b} t={t} first={i === 0} />
      ))}
    </View>
  );
});

const styles = StyleSheet.create({
  inlineCode: { fontFamily: MONO, fontSize: 13 },
  codeBlock: { borderWidth: StyleSheet.hairlineWidth, borderRadius: 8, padding: 10 },
  quote: { borderLeftWidth: 3, paddingLeft: 10 },
});
