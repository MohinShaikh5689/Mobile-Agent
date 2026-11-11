# Mobile Agent

A local-first personal agent for Android and iOS that answers questions across Gmail, Google Drive and Google Calendar. Built with React Native (Expo) and TypeScript.

Everything runs **on the phone**. The agent is a [LangGraph.js](https://github.com/langchain-ai/langgraphjs) graph inside the app, and the model runs locally through [llama.rn](https://github.com/mybigday/llama.rn) (llama.cpp). There is no server, no Python and no laptop. The only network traffic is to Google's APIs and the one-time model download.

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
| `src/core/llm/onDevice.ts` | The on-device model: tool calls through the model's own chat template, grammar-constrained JSON for the grader and planner, embeddings, one call at a time per context. |
| `src/core/llm/models.ts` | The model list: Qwen3-4B-Instruct-2507 (default) and Phi-4-mini for chat, nomic-embed-text v1.5 for embeddings. |
| `src/llm/` | Model downloads (to a `.part` file, renamed when complete) and the single shared runtime. Model memory is freed when the app goes to the background. |
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

### 1. Device and memory

| Platform | What it needs |
| --- | --- |
| iOS | A device with an Apple7 GPU or newer (A14 / iPhone 12 and later) for Metal. 8 GB RAM is recommended for the 4B models. The app requests the **Increased Memory Limit** and **Extended Virtual Addressing** entitlements (see below). Metal does not work in the simulator, so test on a device. |
| Android | arm64 with 8 GB RAM or more for the 4B models. Runs on the CPU, plus the GPU through OpenCL on Qualcomm Adreno chips. |

**iOS entitlements.** `app.json` declares `com.apple.developer.kernel.increased-memory-limit` and `com.apple.developer.kernel.extended-virtual-addressing` for every build, not only production. Enable both capabilities on the App ID (`com.mobileagent.app`) in the Apple Developer portal. EAS Build syncs capabilities automatically; with local Xcode builds, enable them under Signing & Capabilities.

**Models.** On first launch, open **Settings → On-device models** and download one chat model (about 2.5 GB) and the embedding model (about 150 MB), preferably over Wi-Fi. Files go to the app's Documents folder. If a link has moved, use **Edit URL** on that model.

### 2. Google OAuth

The app needs Google OAuth **client IDs**: no API key and no client secret. Phone apps sign in with PKCE, so nothing secret ships in the app. Each user signs in with their own Google account, and the app only ever sees that user's data.

In the [Google Cloud Console](https://console.cloud.google.com/), in one project:

1. **Enable the APIs:** Gmail API, Google Drive API, Google Calendar API.
2. **OAuth consent screen:** choose *External*, leave it in **Testing**, and add every Google account that will sign in as a **test user**. Add the scopes `gmail.readonly`, `drive.readonly` and `calendar.readonly`.
3. **Create the client IDs** (Credentials → Create credentials → OAuth client ID):
   - **iOS:** bundle ID `com.mobileagent.app`. This is the one the IPA needs.
   - **Android:** package `com.mobileagent.app`, plus the SHA-1 of the signing key (`cd android && ./gradlew signingReport` after a prebuild).
4. **Give the IDs to the build:**
   - Locally: copy `.env.example` to `.env` and fill them in.
   - In GitHub Actions: add them as repository **variables** (not secrets) named `EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID` and `EXPO_PUBLIC_GOOGLE_ANDROID_CLIENT_ID`.

The iOS redirect scheme is set up for you: `app.config.ts` derives `com.googleusercontent.apps.<id>` from the iOS client ID and registers it, and the app redirects there. That redirect does not depend on the bundle ID, so it keeps working after a sideloading tool re-signs the app.

**Limits of Testing mode:**
- Only listed test users can sign in, up to 100.
- Google shows an "unverified app" warning; tap *Continue*.
- Refresh tokens expire after **7 days**, so you sign in again weekly.

Removing those limits means publishing the app. `gmail.readonly` is a restricted scope, which then requires Google's verification and a security assessment. For personal use, Testing mode is enough.

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

1. Download the models on the **Settings** tab.
2. Sign in to Google on the **Settings** tab.
3. Tap **Sync now** to build the local index. Embedding runs on the phone, so keep it plugged in for the first sync.
4. Ask questions on the **Chat** tab.

### Unsigned IPA for sideloading (iOS)

The **iOS unsigned IPA** workflow (`.github/workflows/ios-unsigned-ipa.yml`) builds an IPA without an Apple certificate. Run it from the Actions tab, or push a `v*` tag. It runs the tests, builds on a macOS runner, and uploads `MobileAgent-<version>-<sha>-unsigned.ipa` as an artifact. Install it with a sideloading tool such as AltStore, SideStore or Sideloadly, which signs it with your Apple ID.

- **Before the first run:** set the Google client IDs as repository variables (Settings → Secrets and variables → Actions → Variables): `EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID`, plus the Android and web ones if you use them. They are compiled into the app.
- **Entitlements:** the app is ad-hoc signed with its entitlements (no certificate), because sideloading tools re-sign with whatever entitlements the binary carries. Whether iOS honours `increased-memory-limit` then depends on the Apple ID that signs it. If your signing tool or account drops it, large models may be killed for using too much memory. Check the tool's log for the entitlement.
- **Bundle ID:** a free Apple ID usually rewrites the bundle ID when signing. Google sign-in still returns to the app, because it uses the reversed client ID scheme, not the bundle ID.
- **Build time:** the build compiles llama.cpp and takes roughly 30–60 minutes on a GitHub macOS runner.

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
| Phone | Measured memory and speed on target devices, and a 1.7B fallback for phones with less RAM. |
