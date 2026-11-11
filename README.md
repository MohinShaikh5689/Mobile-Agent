# Mobile Agent

A local-first personal agent for Android and iOS that answers questions across Gmail, Google Drive and Google Calendar. Built with React Native (Expo) and TypeScript.

The agent runs **inside the app** as a [LangGraph.js](https://github.com/langchain-ai/langgraphjs) graph. There is no Python and no Node server. The only thing outside the phone is the model, served by Ollama on your laptop.

What is built:
- **LangGraph graph:** router, specialists, a grader with one query rewrite, and a planner. It is checkpointed to on-device SQLite at every step.
- **Rules-based router:** each request goes down one path: exact lookup, recall, multi-step, or action. The router never asks the model.
- **Specialists:** mail, docs and calendar each get two tools.
- **Hybrid retriever:** a local index on the phone answers for older mail and docs. Anything newer is fetched live. Both sets are merged and reranked together.
- **Incremental sync:** uses Gmail history IDs and the Drive changes feed.
- **Tool loop:** has a step limit and an error path for every tool error. Untrusted content is always wrapped as data.
- **Write actions are not built yet.** Requests to draft, send or invite get an explanation instead of an action.

## How it fits together

```
START → router ─┬─ structured ─► specialist ─────────────────────────────────► END
                ├─ fuzzy ──────► retrieve ─► grade ─┬─ relevant ─► respond ─► END
                │                    ▲              └─ rewrite ──┘ (once)
                ├─ multi_step ─► planner ─► run_step ⟲ (one per plan step) ─► compose ─► END
                └─ action ─────► action ────────────────────────────────────► END

retrieve:  local index [floor, watermark)  +  live API before floor and since watermark
```

| Path | What it does |
| --- | --- |
| `src/core/graph/agentGraph.ts` | The LangGraph graph: state, nodes and edges. |
| `src/core/graph/checkpointer.ts` | Checkpointer that writes each thread through to SQLite, so a paused run (the future approval gate) resumes after a restart. |
| `src/polyfills.ts` | Web APIs that LangGraph.js needs and Hermes may lack: streams, TextEncoder/TextDecoder, crypto, `Symbol.asyncIterator` and a full `URL`. Loaded before everything else. |
| `src/core/` | Framework-free agent logic: router, time windows, watermark plan, retriever, tool loop, specialists. Unit-tested. |
| `src/core/watermark.ts` | The watermark rule. The index covers `[floor, watermark)`; anything outside is fetched live. |
| `src/core/retriever.ts` | Hybrid retrieval, merge and rerank, and the weak-result live fallback. |
| `src/core/llm/` | Model interface plus the Ollama backend (chat, schema-constrained JSON, embeddings). An on-device backend can implement the same interface. |
| `src/google/` | Gmail, Drive and Calendar clients, OAuth, and incremental sync. |
| `src/storage/` | SQLite vector index and key-value store on the device. |
| `src/ui/` | Chat and Settings screens. Each answer shows its graph path, plan, retrieval windows, tool calls and sources. |

### Why LangGraph.js and not Python

Python LangGraph cannot ship inside an Android or iOS app; it would need a separate server. LangGraph.js has the same model (state, nodes, conditional edges, checkpointers, breakpoints) and runs in the app's JavaScript engine. The app uses its `web` entry point, which avoids Node-only modules. Two consequences:

- `interrupt()` inside a node does not work in the `web` build, because it relies on Node's AsyncLocalStorage. Pauses use `interruptBefore` breakpoints instead. The phase 4 approval gate will pause before an `execute` node, then resume after approval. This is tested across a simulated restart.
- Hermes needs the polyfills in `src/polyfills.ts`. To check them, the graph was bundled and run on a standalone Hermes engine. It routes, retrieves, grades, answers, and reloads a checkpoint.

### The watermark, precisely

- **watermark**: when the last sync *started*. The index is complete up to this time. Anything that arrives during a sync is fetched live and de-duplicated.
- **floor**: how far back the first sync reached. The default is 180 days, capped at 300 items per source. Older periods are fetched live.
- **Weak results**: if the index part returns nothing, or nothing scoring above 0.35, the whole window is queried live before the agent says "not found". After that, the graph's grader can rewrite the query once.
- **Reranking**: live results are embedded with the same model and scored against the question exactly like index hits, so a match ranks the same whichever side of the watermark it falls on.

## Running it

### 1. Model server on the laptop (Ollama)

```bash
ollama pull qwen3:4b-instruct
```

```bash
ollama pull nomic-embed-text
```

```bash
OLLAMA_HOST=0.0.0.0 ollama serve
```

The phone and the laptop must be on the same Wi-Fi network. In the app, open **Settings** and set the Ollama URL to `http://<laptop-LAN-IP>:11434`, then tap **Test connection**. You can change the chat and embedding models there, for example to compare the four candidates in the project doc.

### 2. Google OAuth

In the [Google Cloud Console](https://console.cloud.google.com/):

1. Enable the **Gmail API**, **Google Drive API** and **Google Calendar API**.
2. On the OAuth consent screen, add yourself as a test user. Add the scopes `gmail.readonly`, `drive.readonly` and `calendar.readonly`.
3. Create OAuth client IDs:
   - **iOS**, bundle ID `com.mobileagent.app`.
   - **Android**, package `com.mobileagent.app`, plus the SHA-1 of your debug keystore (`cd android && ./gradlew signingReport` after the first prebuild).
4. Copy `.env.example` to `.env` and fill in the IDs.
5. Register the redirect schemes Google uses. In `app.json`, change `"scheme"` to an array that includes the reversed iOS client ID:
   `["mobileagent", "com.googleusercontent.apps.<your-ios-client-id-prefix>"]`.

Scopes are read-only. They widen only when write actions are built.

### 3. Build and run on a device

Google sign-in needs a development build. Expo Go cannot handle the native OAuth redirect.

```bash
npm install
```

```bash
npx expo run:android
```

```bash
npx expo run:ios
```

`run:ios` needs a Mac with Xcode. To build in the cloud instead, use [EAS Build](https://docs.expo.dev/build/introduction/) (`eas build --profile development`).

Once the app is running:

1. Sign in on the **Settings** tab.
2. Tap **Sync now** to build the local index.
3. Ask questions on the **Chat** tab.

## Development

```bash
npm test
```

```bash
npm run typecheck
```

The core and the graph are tested with fakes: a bag-of-words embedder, a scripted model and fake services. The tests cover:
- Routing for the sample questions in the project doc.
- Date parsing.
- Every case of the watermark plan.
- The "same answer before or after the watermark" check.
- The weak-result fallback.
- Tool-loop error paths and the step limit.
- Every graph path, the grader's rewrite loop, and planner fallback.
- Checkpoints that survive a restart, including resuming a paused run.

## Not built yet

| Phase | Item |
| --- | --- |
| 2 | Background sync while idle or charging (`expo-background-task`). Sync is manual for now. |
| 3 | The eval set (at least 30 tasks, including injection cases) with route accuracy. The graph itself is built. |
| 4 | Write actions: drafts and event changes behind an approval gate (an `interruptBefore` breakpoint) that shows exactly what will happen. Re-fetch by ID before acting. |
| 5 | An MCP server per service, tracing, and a four-model comparison. |
| Phone | An on-device model (for example llama.rn with a 3–4B Q4 GGUF) behind the same `ChatModel` and `Embedder` interfaces, so the laptop is no longer needed. |
