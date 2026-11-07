# StrainlessAI v2 — Specification

Owner: Prem Nadesan · Last updated: 2026-11-06 · Status: draft for build

This is the source of truth for the v2 rebuild. The build guide says *how* to build each part; this file says *what* to build. Part numbers match the build guide. If code and this spec disagree, fix the spec first, then the code.

Items marked **[DEFAULT]** are provisional choices so work isn't blocked. Items marked **[TUNE]** are starting values to adjust with real data. Section 10 lists the open decisions.

---

## 0. Scope

### In v2
- A binary **Healthy vs Strained** model on interpretable acoustic features, shown to users as three calibrated bands: **Healthy / Borderline / Strained**.
- One fixed arpeggio per voice type (4 voice types).
- Recording quality checks before scoring.
- RAG coach rebuilt on heading-matched retrieval in Qdrant 1.20, with gpt-oss on Groq.
- React front end; Flask becomes a JSON API that also serves the built front end.
- Optional voice-type finder.

### Out of v2 (moved to v3)
- Weak / breathy detection.
- Arpeggios fitted to each singer's own range.
- Cross-cultural evaluation and per-tradition reporting (the `tradition` field is still recorded).
- Publication, literature review, IRB.
- Any speaking-voice baseline or loudness ratio.
- EGG or any hardware beyond a phone or laptop mic.

### Removed from v1
Spectrogram CNN, TensorFlow, librosa, matplotlib, LangChain, the Weak class, letter grades (A–F), Llama models, HTML-in-Python template, `ScriptProcessorNode`, fixed 22,050 Hz capture.

### Glossary
| Term | Meaning |
|---|---|
| Take / clip | One recording of the arpeggio. One WAV + one JSON. |
| Singer | One person, identified by `singer_id`. All of a singer's clips stay in the same data split. |
| Condition | What the singer was asked to do for a take (comfortable / pushed / edge). Not the label. |
| Rating | One rater's judgment of a clip: none / mild / clear strain. |
| Label | The consensus of the ratings (Section 2.7). |
| P(strain) | The model's calibrated probability that a clip is strained. |
| Band | Healthy / Borderline / Strained, from P(strain) and the quality checks. |

---

## 1. Project layout and conventions

```
<repo>/
  CLAUDE.md
  docs/        SPEC_v2.md, data_protocol.md, model_card.md, progress.md, rag/, research/
  api/         Flask app: app.py, llm.py, coach.py, quality.py, static/audio/
  ml/          arpeggios.py, collect/, rate/, features/, train/
  rag/         sections.json, paraphrases.json, ingest.py, retrieve.py, eval/
  web/         Vite + React + TypeScript + Tailwind + shadcn/ui
  models/      strain_v2.joblib (committed; small)
  reports/     generated plots and tables (gitignored)
  data/        raw/, labels/, splits.json, features.parquet (gitignored, never committed)
  scripts/     one-off checks (check_llm.py, quality_report.py, compare_v1_v2.py)
```

- Python 3.12, managed with **uv**. Tests with **pytest**. Node LTS for `web/`.
- **Banned in the v2 project:** TensorFlow, librosa, LangChain, matplotlib in the API path (matplotlib is allowed in `ml/` and `scripts/` for reports only).
- **Secrets** live only in `.env` (never committed, never pasted into prompts):
  - `GROQ_API_KEY`, `LLM_MODEL`
  - `QDRANT_CLOUD_URL`, `QDRANT_CLOUD_API_KEY`, `QDRANT_COLLECTION` (default `singing_v2`)
  - `COLLECT_BUCKET` (private Cloud Storage bucket for the collection site), `PAYMENT_FORM_URL` (Prem's Google Form)
- Every finished build step appends one line to `docs/progress.md`: `YYYY-MM-DD · step · result`.
- Recordings, consent forms and anything that identifies a singer never go into git or into any cloud-synced notes.

### 1.1 Every screen works on phone and desktop
This applies to everything people see: the collection site (2.8), the v2 app (8.3) and the trainer.
- **Equally clear at both sizes**, not desktop-only or phone-only. Check every screen at **375 px wide** (phone), **768 px** (tablet) and **1280 px** (laptop). Nothing is cut off, overlaps, or needs sideways scrolling.
- **Touch first:** buttons and tap targets at least 44 × 44 px with room between them. Nothing works only on hover; anything a desktop user hovers for (vocab definitions, help text) also opens with a tap.
- **Readable text:** body text at least 16 px on phones (this also stops iPhones from zooming into form fields), and contrast that meets WCAG AA.
- **Primary actions stay reachable:** on a phone the main button for each screen (Record, Stop, Restart, Submit) is visible without scrolling while it matters.
- **Microphone permission** works and explains itself on iOS Safari, Android Chrome, and desktop Chrome, Edge and Safari. If permission is denied, the screen says how to turn it back on for that browser.
- **Done means tested** on a real iPhone, a real Android phone and a laptop in at least two browsers, not only in the browser's device emulator.

---

## 2. Data collection and rating

### 2.1 The task
- Vowel: "ah" /ɑ/, held plain: no vibrato added on purpose, no ornaments or slides between notes.
- Pattern: major arpeggio 1-3-5-8-5-3-1, one beat per note except the **top note, held for 2 beats** (8 beats in all). Strain shows most on the top notes, so the held top note gives the longest, cleanest sample where it matters.
- Tempo: **90 BPM** in the examples and the count-in (about 5.3 s of singing). Singers won't match it exactly, and nothing downstream depends on exact timing: notes are found by pitch (Section 6.1), and only a minimum note length is enforced (Section 5.3).

| Voice type | Notes | Approx. Hz |
|---|---|---|
| Baritone | F3 A3 C4 F4 (held) C4 A3 F3 | 175 220 262 349 |
| Tenor | Ab3 C4 Eb4 Ab4 (held) Eb4 C4 Ab3 | 208 262 311 415 |
| Alto (contraltos and mezzo-sopranos) | F4 A4 C5 F5 (held) C5 A4 F4 | 349 440 523 698 |
| Soprano | Ab4 C5 Eb5 Ab5 (held) Eb5 C5 Ab4 | 415 523 622 831 |

Basses use the baritone arpeggio.

### 2.2 Example arpeggios
Singers copy what they hear, so the examples set the pitch, tempo, vowel and style of every take. They're needed **before data collection starts**, and they stay the same for all of it.

Four files per set: `tenor.wav`, `baritone.wav`, `alto.wav`, `soprano.wav`. There are two sets.

**Synthesized examples (all of data collection).**
- Generated by `scripts/make_synth_examples.py` into `api/static/audio/`. These are the examples for every collection take, not temporary placeholders, and they aren't changed once collection starts, so every training take copies the same reference.
- Each one: a soft harmonic tone (not a pure sine) at the exact pitches of that voice type's arpeggio from Section 2.1, 90 BPM, top note held 2 beats, no vibrato, no glides between notes. Mono 16-bit WAV, 44.1 kHz, peaks at −6 dBFS, 0.5 s of silence before and after.
- Each must land within ±20 cents of every target note (`scripts/check_examples.py`).
- The collection recorder (Section 2.8) always plays this set. The app plays it too until the switch below passes.

**Sung examples (the v2 app, Step 8a).**
- Recorded later, for the app only. Kept beside the synthesized set, not over it, in `api/static/audio/sung/` [DEFAULT].
- Each one: a healthy, comfortable take of that voice type's arpeggio from Section 2.1, on "ah", notes held plain, top note held 2 beats, no vibrato added on purpose, no accompaniment. Sung by a trained Purdue singer of that voice type, with a 90 BPM metronome click in headphones (so the click isn't recorded). Anything from 85 to 95 BPM is fine; exact tempo isn't required.
- Recorded in a quiet room, mono WAV, peaks around −6 dBFS, about 0.5 s of silence before and after.
- Each must pass `api/quality.py` and land within ±20 cents of every target note (`scripts/check_examples.py`).

**Switching the app to the sung examples.** The model was trained on takes that copied the synthesized examples, so the app switches only if the sung examples don't change what singers produce.
- 3–4 singers each record comfortable takes after hearing each version: 3 takes per version, with the order of the versions alternated between singers [DEFAULT].
- Compare, per singer, P(strain) and the model's main features (its 5 most important, from Section 7) between the two versions.
- The app switches only if they're close: per-singer mean P(strain) differs by ≤ 0.05, and each main feature's mean differs by less than that singer's take-to-take standard deviation after the same example [TUNE]. Otherwise the app keeps the synthesized examples.

### 2.3 Takes per singer (16)
All takes use the singer's own voice-type arpeggio. **Pitch never changes between conditions; only effort does.** Takes are recorded in the order below (comfortable, then edge, then pushed) so pushed takes can't tire the voice before the comfortable ones. Every pushed take shows: "Stop if anything hurts or feels wrong. A few strained takes are enough; never push through pain."

| Condition | Takes | Instruction to the singer |
|---|---|---|
| comfortable | 6 (slots 1–6) | Easy, healthy production. Slots 5–6 sung loudly but freely (healthy belt). |
| edge | 4 (slots 7–10) | In between: a little more effort than comfortable, not fully pushed. |
| pushed | 6 (slots 11–16) | Same notes with excess effort: squeezed, pressed, "shouty" throat. Slots 15–16 at moderate volume, so strain isn't the same as loudness. |

### 2.4 Targets
| | Minimum viable | Solid v2 |
|---|---|---|
| Distinct singers | 30 | 50–60 |
| Clips per singer | 16 | 16 |
| Total clips | ~480 | ~800–960 |
| Held-out test singers | 8–10 | 12–15 |

Balance sex and voice type as far as recruitment allows. Raters decide labels, so the healthy / mild / strained counts won't match the conditions; check counts after each rating batch.

### 2.5 Consent and personal data
- Singers are 18 or older (post-puberty voices, and able to consent for themselves).
- Consent is given on the collection site (Section 2.8): the consent text (kept in `docs/data_protocol.md`, currently `consent-v1`; any wording change gets a new version), an "I am 18 or older" checkbox, and the singer's typed full name as a signature. The site stores the consent record (singer_id, typed name, consent text version, timestamp) under `consent/` in the private bucket, **separate from the recordings**. Recording metadata never contains a name.
- **The site never stores email addresses or payment details.**
  - Emails and the email → singer_id mapping live only in Prem's private recruiting sheet.
  - Payment handles (Venmo/Zelle) live only in the responses of Prem's Google Form. Delete each response once that singer is paid.
- Singers may withdraw: their recordings, consent record and invite are deleted from the bucket and `data/`, and they're left out of the next training run.

### 2.6 Clip metadata (`data/raw/<clip_id>.json`)
| Field | Type | Required | Example |
|---|---|---|---|
| clip_id | string | yes | `c0042` |
| singer_id | string | yes | `s007` |
| session_id | string | yes | `s007-1` |
| recorded_at | ISO 8601 string | yes | `2026-11-20T14:05:00-05:00` |
| voice_type | `tenor` \| `baritone` \| `alto` \| `soprano` | yes | `alto` |
| arpeggio_id | string: `<voice_type>_<lowest note>` (`baritone_F3`, `tenor_Ab3`, `alto_F4`, `soprano_Ab4`) | yes | `alto_F4` |
| condition | `comfortable` \| `pushed` \| `edge` | yes | `pushed` |
| slot | int 1–16 | yes | `12` |
| loud | bool | yes | `true` (singer was asked to sing loudly: slots 5–6 only) |
| sex | `F` \| `M` \| `unknown` | yes | `unknown` |
| tradition | free text | no | `Carnatic` (recorded for v3, unused in v2) |
| device | `phone` \| `laptop` \| `tablet` \| `desktop`, plus mic `built-in` \| `wired headset` \| `USB mic` \| `other` (from a dropdown) | yes | `phone / built-in` |
| user_agent | string | yes | browser UA string |
| track_settings | object | yes | output of `track.getSettings()` |
| sample_rate | int | yes | `48000` |
| duration_s | float | yes | `9.8` |
| finder_low, finder_high | note name | no | `E3`, `A5` |

The WAV is mono, 16-bit PCM, at the device's native sample rate.

### 2.7 Rating and labels
- Raters: Prem and a voice teacher, both labeling every clip. Prem recorded the singers, so he may recognize voices; that's acceptable, but the voice teacher's ratings settle any doubt about the rating definitions.
- Every clip is rated blind: the rating page shows no singer, condition, voice type or tradition, and plays clips in random order.
- Ratings: `data/labels/ratings_<rater_id>.csv` with `clip_id, rater_id, rating (none|mild|clear), rated_at`.
- **Consensus** (`data/labels/consensus.csv`: `clip_id, label, ratings`):

| Ratings | Label |
|---|---|
| All raters agree | that rating (none → `healthy`, mild → `mild`, clear → `strained`) |
| Split by one step (none + mild, or mild + clear) | `mild` |
| none + clear (2 raters) | `excluded`, unless a third rater breaks the tie |
| 3+ raters, no agreement | majority; no majority → `mild` |

- Agreement: weighted Cohen's kappa between each pair of raters, computed after the first 3 singers and after every batch. Below **0.6**, raters re-align on definitions and re-rate before more recording. [TUNE]

### 2.8 Collection site
A small private website where invited singers record their 16 takes on their own device, on their own time. It's a **separate Cloud Run service** (`strainless-collect`) from the v2 app, so collection can run before the app exists.

**Stack:** Flask + one plain HTML/JavaScript page (no React needed). Every screen meets Section 1.1: singers will use phones and laptops about equally. It imports `api/quality.py` and `ml/features/segment.py`, so Part 5 and Step 6a must be finished before it goes live.

**Invites**
- `ml/collect/make_invites.py N` creates N new `singer_id`s, each with a random URL-safe token (at least 16 random bytes). It writes `invites.csv` (`singer_id, link`) to Prem's machine only (never git) and stores a hash of each token → singer_id in the bucket under `invites/`.
- Prem emails each recruit their personal link (`https://<collect-url>/s/<token>`). The token is the only login. An invite can be revoked by deleting it; a link stops working 30 days after it's created. [TUNE]

**Singer flow**
1. **Welcome:** what the session involves (about 20–30 minutes, 16 short takes, can stop and come back), and that pay ($10) follows completion.
2. **Consent:** Section 2.5.
3. **About you:** voice type (four cards with range and a play button for each example), sex (optional), device and mic (dropdowns), tradition (optional, free text).
4. **Mic check:** a level meter, then a short test take checked for clipping, noise and bandwidth (Section 5.3). Failures say what to fix (move back, quieter room, don't use Bluetooth earbuds) before any real take.
5. **Session board:** 16 slots in order, grouped by condition, each showing its instruction and status: *empty*, *checking*, *accepted* (✓) or *rejected* (with the reason).
   - Tapping a slot opens it. The singer can **play back** an accepted take, **re-record** it (the new take replaces the old one once it's accepted), or **delete** it (the slot goes back to empty).
   - **While recording, a Restart button** throws away the take in progress and starts the count-in again. Nothing is uploaded.
   - Recording follows Section 5.2. When the singer stops, the take uploads and runs the Section 5.3 checks; the result appears within a few seconds. A rejected take is discarded (not stored) with its reason shown, and the slot stays open for another try.
   - Progress is saved on the server after every accepted take. Opening the link again returns to the board.
6. **Submit:** enabled only when all 16 slots are accepted. A confirmation says takes can't be changed afterward. Submitting locks the session and shows a **completion code**: 8 characters from an unambiguous alphabet (no 0/O, 1/I/L), unique per singer, stored with the session. The screen shows the code and a button to the payment Google Form (URL from `PAYMENT_FORM_URL`).

**Payment (outside the site)**
- Prem's Google Form asks for the completion code and a Venmo username or Zelle phone/email.
- Prem matches each code with `ml/collect/report.py` (submitted sessions and their codes), pays $10, then deletes that form response.

**Storage (private bucket, `COLLECT_BUCKET`)**
| Prefix | Contents |
|---|---|
| `invites/` | token hash → singer_id, created_at, revoked |
| `consent/<singer_id>.json` | consent record (Section 2.5) |
| `sessions/<singer_id>.json` | status (`open` / `submitted`), slot statuses, completion code, rejected-take count per slot |
| `takes/<singer_id>/<clip_id>.wav` and `.json` | accepted takes only, with Section 2.6 metadata |

**Tools on Prem's machine**
- `ml/collect/report.py`: one line per session (singer_id, accepted slots, submitted, completion code).
- `ml/collect/pull.py`: downloads every submitted session's takes into `data/raw/`. Only submitted sessions are pulled, never open ones.

**Security and limits:** HTTPS (Cloud Run), private bucket written only by the service's own identity, uploads limited to WAV under 16 MB, 30 uploads per singer per hour (enough for re-records), no public listing of anything.

**Cost:** Google Cloud billing must be on, with a $10/month budget alert set first. Cloud Run scales to zero and storage for ~1,000 short WAVs is a few GB, so expect well under a dollar a month. Singer payments ($10 each) are the real cost.

---

## 3. LLM

- Provider: Groq. Model from `LLM_MODEL`, default `openai/gpt-oss-120b`; fallback `openai/gpt-oss-20b`.
- Reasoning effort: low. Timeout 25 s, 1 retry. Completion limit 1024 tokens; check whether reasoning tokens count toward it and raise the limit if replies get cut off. [TUNE]
- All LLM calls go through `api/llm.py: chat(messages) -> str`.
- On failure, `/chat` returns a friendly fallback message (HTTP 200, `"error": "llm_unavailable"`). **Voice analysis never depends on the LLM** (coaching for results is static text, Section 4.6).

---

## 4. RAG coach

### 4.1 Source
`docs/rag/Singing_Rag_Doc_V2.pdf` → `rag/sections.json`, converted once, then edited by hand. The PDF is no longer read after conversion.

```json
[{"id": "s19", "title": "High notes feel like shouting", "body": "…"}]
```
Exactly 62 sections. `id` = `s` + 2-digit order in the PDF.

### 4.2 Paraphrases
`rag/paraphrases.json`: `{"s19": ["why do I sound like I'm yelling on high notes", …]}`, 5 per section, generated with the LLM, reviewed by hand. The generator never sees the eval set.

### 4.3 Collection (`singing_v2`)
- One point per title and one per paraphrase (62 × 6 = 372 points).
- Payload: `{parent_id, text, title, body, kind: "title"|"paraphrase"}`.
- Dense vector: `BAAI/bge-large-en-v1.5` via fastembed, cosine. Queries get the bge query prefix; stored texts don't.
- Sparse: BM25 on `text`, using Qdrant 1.20 text-index scoring if it works inside a hybrid prefetch; otherwise fastembed BM25 sparse vectors. Record which in `docs/progress.md`.
- `rag/ingest.py` recreates the collection from scratch on every run.

### 4.4 Retrieval (`rag/retrieve.py`)
1. Dense and BM25 prefetch, fused with RRF.
2. Group by `parent_id`, one hit per section.
3. If the top fused score < `THRESHOLD` → return nothing.
4. Return the top section, plus the second if its score is within `MARGIN` of the top.
5. `THRESHOLD` and `MARGIN` are chosen by `rag/eval/run_eval.py` and stored in `rag/config.json`.

### 4.5 Eval
- `rag/eval/queries.csv`: `query, expected_id` (blank for off-topic). 60–100 queries written by a person, about 15 off-topic.
- Metrics: hit@1, hit@3, off-topic rejection rate, for v1 (`singing_chatbot`, k=1) and v2 side by side.
- **Acceptance:** v2 hit@1 > v1 hit@1, and ≥ 90% of off-topic queries get no context. [TUNE]

### 4.6 Coach (`api/coach.py`)
- `answer(message, history) -> str`: retrieve → system prompt (friendly voice teacher, the vocab definitions, the retrieved section body if any) → last 6 history messages → user message → `llm.chat()`. Markdown allowed.
- Vocab definitions: the 28-term list from the current v1 `app.py` on `main`.
- `coaching_for(band) -> list[{id, title, body}]`, **no LLM call**:
  - Strained → sections titled "High notes feel like shouting", "Too much pressure builds in the throat", "Singing high notes takes too much effort".
  - Borderline meaning "some strain" → only "Too much pressure builds in the throat".
  - Borderline meaning "unclear result" → no coaching (the UI asks for a re-record).
  - Healthy → none.
  - Titles are resolved to ids at startup; a missing title stops the app with an error.

---

## 5. Audio capture and quality checks

### 5.1 Browser capture (same in the collection recorder and the app)
| Setting | Value |
|---|---|
| echoCancellation, noiseSuppression, autoGainControl | false |
| Sample rate | device native; never resampled in the browser |
| Channels | 1 |
| Capture | AudioWorklet |
| Encoding | WAV, 16-bit PCM, native rate |
| Logged | `track.getSettings()` sent with each upload |

### 5.2 Recording flow
1. Before the first take of a session, the app plays the voice type's full example arpeggio (Section 2.2). A "Hear example" button replays it any time before recording.
2. User taps Record. The app plays the starting note (1 s) and a 3-beat count-in at 90 BPM, through the speaker.
3. Playback stops. 200 ms later recording starts.
4. **1 s pre-roll:** the screen says "Wait…"; this silence measures the room's noise floor.
5. Screen says "Sing now". The singer sings the arpeggio.
6. Recording stops when the user taps Stop, after 1.5 s of silence once singing has started, or at a hard cap of 15 s, whichever comes first. This covers singers who go slower than 90 BPM. [TUNE]

Nothing plays while recording. Headphones are not required.

### 5.3 Quality checks (`api/quality.py`)
All thresholds live in one `QUALITY_CONFIG` dict in `api/quality.py` (the only place they're defined).

| Check | Measure | Reject if | Flag if |
|---|---|---|---|
| clipping | share of samples with \|x\| ≥ 0.99 | > 0.1% [TUNE] | — |
| noise | SNR = voiced-note RMS vs pre-roll RMS | < 15 dB [TUNE] | 15–25 dB [TUNE] |
| bandwidth | `sample_rate` from track_settings, plus energy above 8.5 kHz relative to 1–4 kHz in voiced frames (a sharp cliff, not a gradual roll-off) | sample_rate ≤ 16 kHz, or a cliff at ≤ 8 kHz (Bluetooth call mics) [TUNE] | — |
| completeness | notes found by segmentation | < 7 notes, interval pattern off by > 1 semitone, or key > 2 semitones from target | — |
| note length | steady core of each note | any core < 250 ms ("Slow down a little and hold each note") [TUNE] | — |
| pre-roll | voice detected in the pre-roll | — | noise unmeasured (skip the noise check) |

- Reject → no score; the user sees a specific re-record message.
- Noise flag → the band is shown as Borderline whatever the score, with "Recording was noisy, so this result is less certain." [DEFAULT]

---

## 6. Feature pipeline

### 6.1 Segmentation (`ml/features/segment.py`)
- Arpeggio definitions (note names, beats per note, intervals, 90 BPM) have one home: `ml/arpeggios.py`. `scripts/make_synth_examples.py`, `scripts/check_examples.py` and the pipeline all import them from there.
- Pitch tracking with parselmouth (Praat), floor 60 Hz, ceiling 1100 Hz, 10 ms steps.
- Brief tracker octave errors (a few frames about ±12 semitones off the surrounding pitch) are folded back before segmentation. A whole take sung an octave off is **not** corrected: it shows up as `key_offset_st` ≈ ±12 and `api/quality.py` rejects it.
- Split into the 7 arpeggio notes **by pitch, not by timing**, so any tempo works as long as each note is long enough. Notes are found as stable pitch levels, then lined up against the 1-3-5-8-5-3-1 pattern at whatever key the singer used. A skipped note keeps the others' indices (skip note 6 → indices 1, 2, 3, 4, 5, 7); extra sounds that don't fit the pattern (including the middle of a slow slide, which can hold a level of its own) are dropped. Between notes sung through without a break, the boundary is where the pitch first gets nearer the next note than the current one. Segmentation only reports what it found; `api/quality.py` decides whether a take is complete.
- Keep the steady core of each note: drop the first and last 15% of the note, smooth the pitch over about 200 ms (at least one vibrato cycle at 5.5 Hz, so natural vibrato doesn't count as a glide), then keep the longest unbroken stretch within 0.5 semitone of the note's median. Anything outside that (scoops, slides, drift) is a glide and is cut. [TUNE]
- Output per note: index (1–7; note 4 is the held top note), `start` and `end` of the steady core (used by Section 6.2 and the note-length check in 5.3), `note_start` and `note_end` of the whole note, median F0 of the core (Hz; the median of the ~200 ms-smoothed pitch, i.e. the centre of any vibrato, since a raw median is biased when the core isn't a whole number of vibrato cycles), and that F0 in semitones relative to the **target** root of the voice type's arpeggio (so a take sung 3 semitones high reads about +3 on note 1).
- Per clip: `key_offset_st`, the median difference between the sung notes and the target pattern in semitones (used by the key check in 5.3), and `tempo_bpm`, the singer's measured tempo: the median of onset-to-onset time ÷ beats over pairs of consecutive notes (the top note counts 2 beats; pairs across a skipped note aren't used). `tempo_bpm` is stored for diagnostics and is **not** a model feature.
- `ml/features/segment.py` never imports matplotlib (`api/quality.py` imports it). Diagnostic plots of the note boundaries come from `ml/features/plot_segments.py`.

### 6.2 Per-note features (`ml/features/extract.py`)
Analysis at 22,050 Hz unless stated; the original is downsampled with a proper anti-alias filter (scipy `resample_poly`).

| Group | Feature | Definition / tool |
|---|---|---|
| Tilt | alpha_ratio | eGeMAPS (openSMILE), energy 50–1000 Hz vs 1–5 kHz |
| Tilt | hammarberg | eGeMAPS, peak 0–2 kHz vs 2–5 kHz |
| Tilt | mfcc1, mfcc2 | eGeMAPS means |
| Tilt | spectral_centroid | numpy, power-weighted mean frequency of voiced frames |
| Tilt | lh_ratio | numpy, energy < 4 kHz vs ≥ 4 kHz, dB |
| Tilt | sharpness | Zwicker sharpness (DIN 45692) via the MoSQITo library, on the native-rate original. Uncalibrated mic, so used as a relative value only |
| Cepstral | cpps | Praat CPPS via parselmouth; peak-search range widened to cover 60–1000 Hz |
| Cepstral | r1_amp, r1_prom | real cepstrum of the steady core; amplitude of the first rahmonic at quefrency 1/F0, and its height above a linear regression of the cepstrum (same method as CPP) |
| Cepstral | r2_r1 | second rahmonic amplitude (quefrency 2/F0) ÷ first |
| Periodicity | hnr | Praat harmonicity (cc) |
| Periodicity | jitter_local, shimmer_local | Praat, on the steady core |
| Loudness | loudness | eGeMAPS loudness mean. Computed, **excluded by default** (Section 7.3) |

### 6.3 Per-clip summary
For each per-note feature: **mean**, **max**, **slope** against note F0 in semitones (least squares over the 7 notes), and **top** (the value on the held top note, note 4). About 55 numbers per clip, plus covariates:
- `voice_type` (one-hot), `sex` (`F`/`M`/`unknown`, one-hot), `median_f0_st` (the clip's median F0 relative to the target root, to absorb off-key singing).

### 6.4 Output
`data/features.parquet`: one row per clip that passed quality checks, with `clip_id, singer_id`, all features, covariates, `label`, `condition`, `loud`, and quality flags. Clips labeled `excluded` are kept in the file but never used for training or evaluation.

---

## 7. Model, calibration and bands

### 7.1 Split (`data/splits.json`)
- 8–10 held-out test singers (12–15 at the larger target), balanced by sex and voice type. Fixed once; never changed.
- Code must refuse to read test singers unless run with `--final`.

### 7.2 Training
- Training rows: training singers, label `healthy` (y=0) or `strained` (y=1). `mild` rows are never trained on.
- Candidates: logistic regression (L2), SVM (RBF), LightGBM.
- Validation: leave-one-singer-out. Scaling and recursive feature elimination happen **inside** each fold. Keep at most about 15 features. [TUNE]
- Primary metric: balanced accuracy; tie-break ROC AUC. If two models are within 1 point, choose the simpler one.
- Save out-of-fold P(strain) for every training clip, including `mild`, to `reports/oof.parquet`.

### 7.3 Ablations and learning curve
- Ablations: all features; + loudness; − rahmonic features (r1_amp, r1_prom, r2_r1); − covariates.
- **Loudness stays out** unless adding it improves balanced accuracy by ≥ 2 points. Rahmonics stay in only if removing them costs ≥ 1 point. [TUNE]
- Learning curve at 10, 15, 20 and all training singers. Still clearly rising → record more singers before Step 7d.

### 7.4 Calibration
- Fit on the out-of-fold probabilities, applied to the final model (trained on all training singers).
- Use Platt scaling if there are fewer than about 500 training clips; otherwise compare Platt and isotonic and keep the lower Brier score.

### 7.5 Bands
- `low` = the calibrated P(strain) below which 90% of `healthy` out-of-fold clips fall.
- `high` = the value above which 90% of `strained` out-of-fold clips fall.
- If `low < high`: Healthy < low ≤ Borderline < high ≤ Strained.
- If `low ≥ high` (classes separate cleanly): no Borderline from the score; a single cutoff at the P(strain) that maximizes out-of-fold balanced accuracy.
- **Meaning of Borderline**, decided from the out-of-fold `mild` clips:
  - ≥ 50% of mild clips fall in the band **and** Spearman(P(strain), none/mild/clear) ≥ 0.7 → "some strain".
  - Otherwise → "unclear result, try recording again".
  - Record the decision in `docs/model_card.md` and in the model file.

### 7.6 Model file (`models/strain_v2.joblib`)
A dict: `model, feature_names, calibrator, low, high, borderline_meaning ("some_strain"|"unclear"), version ("strain_v2.0"), trained_at, n_singers, n_clips`.
Quality thresholds are **not** stored here (they live only in `api/quality.py`).

### 7.7 Final test
Run once, with `--final`, on test singers. Report balanced accuracy, ROC AUC, the band distribution for healthy / mild / strained clips, overall and per voice type and sex. Any change after this means recruiting new test singers before quoting new test numbers.

---

## 8. API and UI

### 8.1 API (Flask, same origin as the web app)

**POST `/analyze_voice`** — multipart form
- `audio`: WAV file (≤ 16 MB)
- `voice_type`: required; `sex`: optional; `finder_low`, `finder_high`: optional; `track_settings`: JSON string
- `consent_storage`: `true`/`false`, default `false` (Section 8.2)

Response 200:
```json
{
  "status": "ok",
  "band": "healthy | borderline | strained",
  "borderline_meaning": "some_strain | unclear | null",
  "p_strain": 0.42,
  "quality": {"ok": true, "flags": ["noise_moderate"], "message": null},
  "coaching": [{"id": "s19", "title": "…", "body": "…"}],
  "model_version": "strain_v2.0",
  "saved_clip_id": "u-7f3a9c"  // only when consent_storage was true, else null
}
```
A rejected take is also HTTP 200, with `"status": "rejected"`, `band` and `p_strain` null, and `quality.message` set. HTTP 400 for a malformed request, 500 for server errors.

**POST `/chat`** — JSON `{"message": str (≤ 1000 chars), "history": [{"role": "user"|"assistant", "content": str}] (≤ 6)}` → `{"response": str}`. Rate limit 20 requests per IP per hour (flask-limiter). [DEFAULT]

**GET `/health`** → `{"status": "ok"|"partial", "model_loaded": bool, "rag_loaded": bool, "llm_model": str, "model_version": str}`

**GET `/static/audio/<file>`** → example arpeggios: `tenor.wav`, `baritone.wav`, `alto.wav`, `soprano.wav`.

### 8.2 Privacy
- **User recordings are stored only with explicit, opt-in consent.** The Record screen has an unticked checkbox: "Save my recording to help improve StrainlessAI." Only users who also confirm they are 18 or older can tick it.
- Without consent, audio is processed in memory and discarded.
- With consent: the WAV plus the metadata fields in Section 2.6 (no name, no email, no IP) go to private cloud storage, never to git. Each saved clip gets a random `clip_id`, shown to the user so they can ask for it to be deleted.
- Stored user clips are **not** training data until they've been rated under Section 2.7, and they never enter the held-out test set.
- A short privacy notice explains what's stored, why, and how to request deletion. Update it before launch.
- Server logs keep: timestamp, band, p_strain, quality flags, voice_type, track_settings. No audio, no IP beyond what rate limiting needs.

### 8.3 Screens
1. **Home:** what the check does, one-line disclaimer, start button.
2. **Voice type:** four cards (name, range in note names, play button for the example). "Not sure?" link → finder.
3. **Finder (optional):** sex (optional, "prefer not to say"); slide down to the lowest note and hold 1 s; slide up to the highest full-voice note and hold 1 s. Show the closest type plus a neighbor. Typical ranges [TUNE]: soprano C4–C6, alto (contralto/mezzo) F3–F5, tenor C3–C5, baritone (and bass) A2–A4. Store the raw notes.
4. **Record:** level meter, the flow in Section 5.2.
5. **Result:** band, plain-language explanation, quality message if any, coaching sections. **No letter grades, no raw percentages.**
6. **Chat:** markdown rendering, vocab tooltips, last 6 messages kept in the page only.
7. **Trainer:** v1 `trainer.html` ported to `/trainer`.

### 8.4 Copy rules
- Disclaimer on Home and Result: "StrainlessAI gives feedback on singing technique. It is not a medical diagnosis. If you have pain, hoarseness lasting more than two weeks, or loss of voice, see a doctor."
- The check covers one arpeggio that tops out near the upper-middle of each voice type. The Result screen says so: a Healthy result doesn't mean the singer never strains higher up.
- Keep the Google Analytics tag from v1.

### 8.5 Design
UI UX Pro Max design system in `design-system/strainless/MASTER.md`; components via 21st.dev Magic, within that system. Every screen meets Section 1.1 (phone and desktop).

---

## 9. Deploy and acceptance

- Host: **Google Cloud Run** (same as v1). Flask serves `web/dist` and the API from one origin.
- Container: Python 3.12 + uv, no TensorFlow; includes `web/dist`, `models/strain_v2.joblib`, `rag/config.json`, and the **bge-large embedding model downloaded at build time**. Files written at runtime count against a Cloud Run instance's memory, so nothing large is downloaded on startup.
- Cloud Run settings: **4 GiB memory, 2 vCPU** [TUNE after measuring], startup CPU boost on, minimum instances 0 (accept a slower first request after idle; raise to 1 if cold starts are too slow for users), concurrency 4 [TUNE].
- Consented user recordings go to a private **Cloud Storage** bucket in the same project, written by the service's own identity; no public access.
- Staging first, then production. Keep v1's image and `singing_chatbot` collection for 2 weeks as rollback, then delete.

**v2 ships when all of these hold:**
| Criterion | Target |
|---|---|
| Test-singer balanced accuracy | ≥ v1 on the same clips (v1 runs with its own preprocessing from `main`; its Weak and Strained count as strained) |
| RAG | Section 4.5 acceptance met |
| `/analyze_voice` latency | p95 < 5 s on the production host [TUNE] |
| `/chat` latency | p95 < 6 s [TUNE] |
| Phone + laptop end-to-end | record, result, coaching, chat all work and meet Section 1.1 on a real iPhone, a real Android phone and a laptop |

---

## 10. Decisions and open items

### Decided (2026-11-06)
| Decision | Choice |
|---|---|
| Arpeggios | Baritone F3–F4, tenor Ab3–Ab4, alto F4–F5, soprano Ab4–Ab5; top note held 2 beats |
| Tempo | 90 BPM in examples and count-in; singers aren't required to match it |
| Raters | Prem and a voice teacher |
| Singers | Post-puberty voices only |
| User recordings | Stored only with opt-in consent, 18+ (Section 8.2) |
| v1 baseline | The current `app.py` on `main` (scipy preprocessing, 28-term vocab) |
| Hosting | Google Cloud Run; consented recordings in a private Cloud Storage bucket |
| v1 clips | Not reused; all v2 training data is newly recorded |
| Example arpeggios (changed 2026-11-07) | Synthesized (`scripts/make_synth_examples.py`) for all of data collection. Sung by one trained Purdue singer per voice type (baritone, tenor, alto, soprano) for the v2 app in Step 8a; the app switches to them only if the comparison in Section 2.2 passes |
| Data collection | Remote, through the invite-link collection site (Section 2.8); $10 per completed session, paid through a separate Google Form |

### Still open
| # | Item | Default until decided |
|---|---|---|
| 1 | Cloud Run memory and minimum instances, after measuring | 4 GiB, 2 vCPU, min instances 0 |
| 2 | Noise-flagged takes forced to Borderline | yes |
