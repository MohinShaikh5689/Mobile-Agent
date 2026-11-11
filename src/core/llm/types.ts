/** Model-backend interfaces. The app ships an Ollama backend; an on-device backend can implement the same two. */

export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  toolCalls?: ToolCall[];
  /** For role 'tool': which tool produced this result. */
  toolName?: string;
}

export interface JsonSchema {
  type: 'object';
  properties: Record<string, { type: string; description?: string; enum?: string[] }>;
  required?: string[];
}

export interface ToolSpec {
  name: string;
  description: string;
  parameters: JsonSchema;
}

export interface ChatResult {
  content: string;
  toolCalls: ToolCall[];
}

export interface ChatModel {
  chat(messages: ChatMessage[], tools?: ToolSpec[]): Promise<ChatResult>;
  /**
   * Output constrained to a JSON schema (grammar-constrained where the runtime allows it).
   * Optional: askJson() falls back to parsing plain chat output.
   */
  chatJson?(messages: ChatMessage[], schema: Record<string, unknown>): Promise<unknown>;
}

export interface Embedder {
  embed(texts: string[]): Promise<number[][]>;
}

/** Small models often emit <think> blocks; they are never part of the answer. */
export function stripThinking(text: string): string {
  return text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
}

/** Pulls the first JSON object out of model text (fallback when the backend cannot constrain output). */
export function parseJsonObject(text: string): unknown {
  const clean = stripThinking(text).replace(/```(?:json)?/g, '');
  const start = clean.indexOf('{');
  const end = clean.lastIndexOf('}');
  if (start < 0 || end <= start) throw new Error('Model did not return JSON.');
  return JSON.parse(clean.slice(start, end + 1));
}

/** Structured output from any ChatModel. */
export async function askJson(model: ChatModel, messages: ChatMessage[], schema: Record<string, unknown>): Promise<unknown> {
  if (model.chatJson) return model.chatJson(messages, schema);
  const res = await model.chat([
    ...messages,
    { role: 'user', content: `Reply with only a JSON object matching this schema: ${JSON.stringify(schema)}` },
  ]);
  return parseJsonObject(res.content);
}
