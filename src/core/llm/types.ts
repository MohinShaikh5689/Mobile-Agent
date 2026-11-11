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
}

export interface Embedder {
  embed(texts: string[]): Promise<number[][]>;
}

/** Small models often emit <think> blocks; they are never part of the answer. */
export function stripThinking(text: string): string {
  return text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
}
