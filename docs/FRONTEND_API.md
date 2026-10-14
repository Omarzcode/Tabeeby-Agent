# Tabeeby Chat API: guide for the frontend

Base URL: `https://tabeeby-agent.fastapicloud.dev`
Interactive docs (try requests in the browser): `https://tabeeby-agent.fastapicloud.dev/docs`

Tabeeby is a medical navigator for Egypt. The patient describes symptoms in Arabic
(including Egyptian dialect) or English; Tabeeby replies in the same language,
recommends the right kind of doctor and real Vezeeta doctors with booking links,
and puts an ambulance warning first when symptoms sound like an emergency.

No API key is needed. Calls from a browser on any website are allowed (CORS).

---

## 1. Send a message

`POST /chat` with `Content-Type: application/json`

**First message**: send only the text.

```json
{ "prompt": "ضرسي بيوجعني جامد وانا ساكن في المعادي" }
```

**Every following message**: also send the `session_id` you got back. The server
remembers the conversation (last 10 exchanges, for 6 hours of inactivity).

```json
{ "prompt": "فيه حد أرخص من 300 جنيه؟", "session_id": "9f2c4e8a1b7d3c6e5a0f4b2d" }
```

`prompt`: 1 to 4,000 characters.

### Response (200)

```json
{
  "signal": "llm_generate_success",
  "response": "ألم الضرس الشديد غالبًا يحتاج **دكتور أسنان** ...\n\n**1. Dr. Hedaya Dental Clinic** — Dentistry\n📍 El-Maadi · 💰 200 EGP ...\n🔗 https://www.vezeeta.com/en/dr/...",
  "session_id": "9f2c4e8a1b7d3c6e5a0f4b2d",
  "language": "ar",
  "direction": "rtl",
  "emergency": false,
  "emergency_numbers": null,
  "doctors": [
    {
      "name": "Hedaya  Dental Clinic",
      "specialty": "Dentistry",
      "subspecialties_text": "Pediatric Dentistry, Cosmetic Dentistry, Endodontics, ...",
      "address": "El-Maadi",
      "fee": 200.0,
      "reviews_count": 162,
      "waiting_time_min": 6.0,
      "profile_url": "https://www.vezeeta.com/en/dr/Clinic-Hedaya--Dental--Dentistry"
    }
  ],
  "verified": true,
  "stop_reason": "completed",
  "history": null,
  "total_cost": 0.0075,
  "trace": null
}
```

| Field | What to do with it |
|---|---|
| `response` | The answer text, in **Markdown** (bold, lists, links). Render it as Markdown, and sanitize the HTML (see section 4). |
| `session_id` | Save it (e.g. `localStorage`) and send it with the next message. |
| `language`, `direction` | `"ar"` + `"rtl"` or `"en"` + `"ltr"`. Set `dir` on the message bubble so Arabic displays right-to-left. |
| `emergency` | When `true`, show a red banner above the answer: "Call 123 now". The answer already starts with the warning. |
| `emergency_numbers` | Present when `emergency` is true: `{"Ambulance (Egypt)": "123", "Police": "122", "Fire": "180"}`. Good for tap-to-call buttons. |
| `doctors` | The doctors recommended in the answer, in the same order. Use them to draw doctor cards with a **Book** button linking to `profile_url`. Fields can be missing; show only what is there. Empty list when no doctors were recommended. |
| `verified`, `stop_reason` | Quality info; you can ignore them in the UI. |
| `history` | Only filled in the advanced stateless mode (no `session_id`, sending `chat_history` yourself). |
| `total_cost` | Cost of this message in USD (for information; not needed in the UI). |

Tip: the answer text also contains the doctors as Markdown. If you draw doctor cards
from `doctors`, you can still show the full text: it contains the advice and the
"why this doctor" lines.

## 2. Other endpoints

| Call | Use |
|---|---|
| `GET /chat/{session_id}` | The conversation so far, to redraw the chat after a page reload: `{"session_id": "...", "messages": [{"role": "user" or "assistant", "content": "..."}]}`. Empty `messages` if it expired. |
| `DELETE /chat/{session_id}` | "New chat" button: forget the conversation. Then start again without a `session_id`. |
| `GET /health` | Status check: `{"status": "ok", ...}`. |

`session_id` may only contain letters, digits and `_ - : .` (max 128). Use the one the server gives you.

## 3. Errors

Every error body looks like `{"signal": "...", "message": "..."}` (422 uses FastAPI's own `{"detail": [...]}`).

| Status | Meaning | What the UI should do |
|---|---|---|
| 422 | Bad request: empty `prompt`, more than 4,000 characters, invalid `session_id`. | Validate input before sending. |
| 429 | Too many messages from this visitor (limit: 20 messages per 10 minutes). Header `Retry-After` = seconds to wait. | Show "Please wait N seconds" and disable the send button. |
| 500 | Something failed while answering. | Show "Something went wrong, try again" with a retry button. |
| 503 | The server is starting or not configured. | Retry after a few seconds. |

## 4. Implementation notes

- **Answers take time.** A reply usually takes 5 to 20 seconds (the assistant searches the doctor database and checks its answer), at most about 90 seconds. Show a typing indicator, and use a request timeout of about 100 seconds. There is no streaming.
- **One message at a time** per conversation: disable the send button until the reply arrives.
- **Render Markdown safely.** For example with `marked` + `DOMPurify`:
  `bubble.innerHTML = DOMPurify.sanitize(marked.parse(data.response))`. Open links in a new tab.
- **Right-to-left.** `bubble.dir = data.direction`. For the input box, `dir="auto"` works well for mixed Arabic and English.
- **Medical disclaimer.** Tabeeby does not diagnose or prescribe. Show a small note in the UI, for example: "Tabeeby helps you find the right doctor. It is not a diagnosis. In an emergency call 123."
- **Privacy.** Don't log or store patients' messages in your frontend analytics.

## 5. Minimal example (JavaScript)

```js
const API = "https://tabeeby-agent.fastapicloud.dev";
let sessionId = localStorage.getItem("tabeeby_session");

async function ask(text) {
  const res = await fetch(`${API}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(sessionId ? { prompt: text, session_id: sessionId } : { prompt: text }),
    signal: AbortSignal.timeout(100_000),
  });
  if (res.status === 429) {
    const wait = res.headers.get("Retry-After");
    throw new Error(`Too many messages. Please wait ${wait} seconds.`);
  }
  if (!res.ok) throw new Error("Something went wrong. Please try again.");
  const data = await res.json();
  sessionId = data.session_id;
  localStorage.setItem("tabeeby_session", sessionId);
  return data; // data.response (Markdown), data.direction, data.doctors, data.emergency ...
}

async function newChat() {
  if (sessionId) await fetch(`${API}/chat/${sessionId}`, { method: "DELETE" });
  sessionId = null;
  localStorage.removeItem("tabeeby_session");
}
```

## 6. Good test messages

| Message | Expect |
|---|---|
| `ضرسي بيوجعني جامد وانا ساكن في المعادي` | Arabic, `rtl`, dentists in El-Maadi in `doctors` |
| `I twisted my knee playing football, I live in Nasr City, max 400 EGP` | English, orthopedists in Nasr City, fees ≤ 400 |
| `My father has crushing chest pain spreading to his left arm` | `emergency: true`, answer starts with "call 123" |
| `hi` | A greeting, empty `doctors` |
