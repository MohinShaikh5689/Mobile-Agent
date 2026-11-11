# Mobile Agent

A local-first personal agent for Android and iOS that answers questions across Gmail, Google Drive and Google Calendar. Built with React Native (Expo) and TypeScript.

This is **phase 1 (read-only agent)** of the build plan, plus the **watermark retrieval** from phase 2:

- A rules-based router sends each request down one path: exact lookup, recall, multi-step, or action.
- Specialists for mail, docs and calendar each get two tools.
- A hybrid retriever uses a local index on the phone for older mail and docs, and fetches newer items live. It merges and reranks both.
- Incremental sync uses Gmail history IDs and the Drive changes feed.
- A tool-calling loop with a step limit, an error path for every tool error, and untrusted content wrapped as data.
- Write actions are not built yet. Requests to draft, send or invite get an explanation instead of an action.

## How it fits together

```
question ─► router (rules, no model)
             ├─ exact lookup ─► one specialist + its API tools ─► answer
             ├─ recall ──────► hybrid retriever ─► answer from chunks (re-fetch by id to quote)
             │                   ├─ local index:  [floor, watermark)
             │                   └─ live API:     before floor, and since watermark
             ├─ multi-step ──► each specialist in turn ─► combine   (LangGraph planner in phase 3)
             └─ action ──────► "not built yet"                       (approval gate in phase 4)
```

| Path | What it does |
| --- | --- |
| `src/core/` | Framework-free agent logic: router, time windows, watermark plan, retriever, tool loop, specialists. No React Native imports, fully unit-tested. |
| `src/core/watermark.ts` | The watermark rule. The index covers `[floor, watermark)`. Everything outside that range is fetched live. |
| `src/core/retriever.ts` | Hybrid retrieval, merge and rerank, and the weak-result live fallback. |
| `src/core/llm/` | Model interface plus the Ollama backend. An on-device backend implements the same `ChatModel` and `Embedder`. |
| `src/google/` | Gmail, Drive and Calendar clients, OAuth, and incremental sync. |
| `src/storage/` | SQLite vector index and key-value store on the device. |
| `src/ui/` | Chat and Settings screens. Each answer has a details row showing route, retrieval windows, tool calls and sources. |

### The watermark, precisely

- **watermark**: when the last sync *started*. The index is complete up to this time. Anything that arrives during a sync is fetched live and de-duplicated.
- **floor**: how far back the first sync reached. The default is 180 days, capped at 300 items per source. The index is not complete before this point, so older periods are fetched live too.
- **Weak results**: if the index part returns nothing, or nothing scoring above 0.35, the whole window is queried live before the agent says "not found".
- **Reranking**: live results are embedded with the same model and scored against the question exactly like index hits. A match therefore ranks the same whichever side of the watermark it falls on, which is phase 2's "done when" check (tested in `__tests__/retriever.test.ts`).

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

The core is tested with fakes: a bag-of-words embedder, a scripted model and fake services. The tests cover:
- Routing for the sample questions in the project doc.
- Date parsing.
- Every case of the watermark plan.
- The "same answer before or after the watermark" check.
- The weak-result fallback.
- Tool-loop error paths and the step limit.

## Not built yet

| Phase | Item |
| --- | --- |
| 2 | Background sync while idle or charging (`expo-background-task`). Sync is manual for now. |
| 3 | Graph orchestration (router, specialists, grader, planner) with checkpointing, and the eval set with route accuracy. |
| 4 | Write actions: drafts and event changes behind an approval gate that shows exactly what will happen. Re-fetch by ID before acting. |
| 5 | An MCP server per service, tracing, and a four-model comparison. |
| Phone | An on-device model (for example llama.rn with a 3–4B Q4 GGUF) behind the same `ChatModel` and `Embedder` interfaces, so the laptop is no longer needed. |
