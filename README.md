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
- **Write actions with approval:** drafts, sending, and creating or moving events, each shown as a card you approve, edit or cancel.

## Write actions (with approval)

The assistant can **draft or send email** and **create or move calendar events**. Nothing is written until you approve it:

1. **The request becomes a proposal.** "Reply to that email saying I'll send it Friday" or "Move my 3pm to 4pm" goes to a `propose` step. The model fills in the fields, and code checks them:
   - addresses are real, and names are looked up among people who have emailed you,
   - dates and times parse,
   - an event being changed is one of your real upcoming events.

   If something is missing, the assistant asks instead of guessing.
2. **A card shows exactly what will happen:** recipients, subject and body, or the event's time before and after. You can **Edit** it, tap the action button (*Save draft*, *Send email*, *Create event*, *Update event*), or **Cancel**.
3. **The graph pauses.** It stops before its `execute` step (a LangGraph `interruptBefore` breakpoint), and the pause is saved to the on-device checkpoint. A proposal survives closing the app; its card comes back on reopen.
4. **On approval, `execute` runs.** It receives only the structured fields, never raw email text. It re-fetches the email being replied to, or the event being moved, right before writing.

Drafts are the default for email: "draft", "write" and "reply" save a Gmail draft, and only "send" sends. Cancelling, declining and deleting are not supported yet.

**Setup:** in Google Cloud, add the scopes `https://www.googleapis.com/auth/gmail.compose` and `https://www.googleapis.com/auth/calendar.events` under *Google Auth Platform → Data access*. Then sign out and sign in again in the app and allow the new permissions. Settings shows which permissions Google granted.

## General conversation

Not everything is about your data. Greetings and small talk go straight to the model; so do general questions ("explain how a VPN works"). When the rules cannot tell whether a message is about your mail, docs or calendar, a `classify` step asks the model one yes/no question and routes it to a search or to plain chat. General answers use the conversation history and stream like the rest, with no search or tools. The model has no web access, and says so for recent news.

## Conversation memory

The assistant remembers the open conversation, across app restarts, until you tap **New chat**. It fits any context size:

- **Recent turns** stay word for word; **older turns** are folded into a short running summary by the model.
- **Items already discussed** (emails, docs, events) are remembered by id, so "that email" can be re-read exactly.
- **Follow-ups are rewritten first.** The graph's first node turns "what did she say after that?" into a standalone question, so routing and search work on it. Each answer's details show how it was understood.
- **History has a fixed share** of the context, about a fifth: roughly 900 tokens at 4K, 3.6K at 16K, capped at 6K.

## How it fits together

```
START → contextualize → router ─┬─ structured ─► specialist ─────────────────────────────────► END
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

**Models.** On first launch, open **Settings → On-device models**, pick a chat model, and download it together with the embedding model (about 150 MB), preferably over Wi-Fi. Files go to the app's Documents folder.

| Chat model | Size | Max context offered | Typical fit |
| --- | --- | --- | --- |
| Qwen3 1.7B | 1.1 GB | 32K | Phones with 6 GB RAM or less |
| Qwen3 4B Instruct 2507 (default) | 2.5 GB | 32K | 8 GB phones (16K context on an iPhone 17) |
| Phi-4-mini 3.8B | 2.5 GB | 32K | 8 GB phones |
| Qwen3 8B | 5.0 GB | 32K | 12 GB phones (16K); tight on 8 GB |

**Context** (how much the model reads at once) is set per model: 4K, 8K, 16K or 32K. By default the app uses the largest context that fits the phone comfortably, and Settings shows the memory each option needs. A longer context costs memory (the KV cache grows with it) and makes a long prompt slower to start answering.

The app reads the phone's RAM and labels each model *Fits*, *Tight* or *Too big*. The estimate covers weights, the KV cache (stored at 8-bit, falling back to 16-bit if the device rejects it) and runtime overhead. On a fresh install it preselects the largest model that fits comfortably. You can still download a *Tight* model after a warning; iOS may close the app if memory runs short. **Add a model by link** takes any direct `.gguf` link. If a built-in link has moved, use **Edit URL** on that model.

### 2. Google OAuth

The app needs Google OAuth **client IDs**: no API key and no client secret. Phone apps sign in with PKCE, so nothing secret ships in the app. Each user signs in with their own Google account, and the app only ever sees that user's data.

In the [Google Cloud Console](https://console.cloud.google.com/), in one project:

1. **Enable the APIs:** Gmail API, Google Drive API, Google Calendar API.
2. **OAuth consent screen:** choose *External*, leave it in **Testing**, and add every Google account that will sign in as a **test user**. Add the scopes `gmail.readonly`, `drive.readonly` and `calendar.readonly`.
3. **Create the client IDs** (Credentials → Create credentials → OAuth client ID):
   - **iOS:** bundle ID `com.mobileagent.app`. This is the one the IPA needs.
   - **Android:** package `com.mobileagent.app`, plus the **SHA-1 of your release keystore** (see *Android APK* below; the APK workflow prints it). Then open the client's **Advanced settings** and turn on **Enable custom URI scheme**: the app redirects to `com.mobileagent.app:/oauthredirect`, and Google turns custom schemes off for new Android clients.
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

### Android APK

The **Android APK** workflow (`.github/workflows/android-apk.yml`) builds an arm64 release APK, signed with your own keystore, and uploads it as an artifact. Install it on the phone directly; allow installs from your browser or file manager when Android asks.

Google identifies the Android app by its package name plus the SHA-1 of the key that signs it. So create one keystore and **keep it**: an APK signed with a different key cannot sign in.

1. Create the keystore once, on any machine with Java:

   ```bash
   keytool -genkeypair -v -keystore mobile-agent.keystore -alias mobile-agent -keyalg RSA -keysize 2048 -validity 10000
   ```

2. Print its SHA-1 and put it in the Google Cloud **Android** OAuth client:

   ```bash
   keytool -list -v -keystore mobile-agent.keystore -alias mobile-agent | grep SHA1
   ```

3. Add repository **secrets**:
   - `ANDROID_KEYSTORE_BASE64`: the keystore as base64 (`base64 -w0 mobile-agent.keystore`; on macOS, `base64 -i mobile-agent.keystore`).
   - `ANDROID_KEYSTORE_PASSWORD` and `ANDROID_KEY_PASSWORD`: the passwords you chose.
   - `ANDROID_KEY_ALIAS`: `mobile-agent`.
4. Add the repository **variable** `EXPO_PUBLIC_GOOGLE_ANDROID_CLIENT_ID` with the Android client ID.

`plugins/withReleaseSigning.js` makes release builds use that keystore. Without the secrets, the workflow still builds, but signs with React Native's public debug key and warns. Like the iOS build, it compiles llama.cpp and takes a while.

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
| 5 | An MCP server per service, tracing, and a four-model comparison. |
| Phone | Measured memory and speed on target devices, and a 1.7B fallback for phones with less RAM. |
