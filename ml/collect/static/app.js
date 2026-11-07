// StrainlessAI collection site: one page, plain JS (spec 2.8, capture 5.1, flow 5.2).
"use strict";

const TOKEN = decodeURIComponent(location.pathname.split("/")[2] || "");
const $ = (id) => document.getElementById(id);
const VOICE_ORDER = ["soprano", "alto", "tenor", "baritone"];
const VOICE_LABEL = {
  soprano: "Soprano",
  alto: "Alto (contralto, mezzo-soprano)",
  tenor: "Tenor",
  baritone: "Baritone (and bass)",
};
const COND_LABEL = { comfortable: "Comfortable", edge: "Edge", pushed: "Pushed" };

let S = null; // /api/state payload: slots, flow, arpeggios, consent, options
let session = null;
let chosenVoice = null;
let openSlot = null; // slot number shown in the panel
let checkingSlot = null; // slot whose take is uploading
let slotResult = null; // {accepted, message, notes} for the open panel, after an upload
let micResult = null;
let micReady = false;
let exampleAudio = null;
let playbackUrl = null;

// ------------------------------------------------------------------ API

async function api(method, path, body) {
  const opts = { method, headers: { "X-Invite-Token": TOKEN } };
  if (body instanceof FormData) opts.body = body;
  else if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  let r;
  try {
    r = await fetch(path, opts);
  } catch (e) {
    return { ok: false, status: 0, data: { message: "No connection. Check your internet and try again." } };
  }
  let data = {};
  try {
    data = await r.json();
  } catch (e) {
    /* non-JSON body */
  }
  if (r.status === 403) showError(data.message || "This link isn't valid.");
  if (data.session) session = data.session;
  return { ok: r.ok, status: r.status, data };
}

// ------------------------------------------------------------------ screens

function show(name) {
  document.querySelectorAll(".screen").forEach((el) => (el.hidden = el.dataset.screen !== name));
  window.scrollTo(0, 0);
}

function showError(text) {
  closePanel(true);
  $("error-text").textContent = text;
  $("progress").hidden = true;
  show("error");
}

function acceptedCount() {
  return Object.values(session.slots).filter((v) => v.status === "accepted").length;
}

function updateProgress() {
  const p = $("progress");
  p.hidden = !session || !session.profile;
  if (session) p.textContent = `${acceptedCount()} / ${S.slots.length} takes`;
}

function route() {
  updateProgress();
  if (session.status === "submitted") return showDone(session.completion_code, S.payment_form_url);
  if (session.consent_version !== S.consent.version) return show("welcome");
  if (!session.profile) return showAbout();
  if (!session.mic_check) return showMic();
  showBoard();
}

async function load() {
  if (!TOKEN) return showError("This link isn't complete. Open the full link from your invite email.");
  const r = await api("GET", "/api/state");
  if (r.status === 403) return;
  if (!r.ok) return showError(r.data.message || "Something went wrong. Reload the page to try again.");
  S = r.data;
  session = S.session;
  $("consent-text").innerHTML = S.consent.html; // server-escaped text from docs/data_protocol.md
  route();
}

// ------------------------------------------------------------------ consent

function consentValid() {
  return $("age-ok").checked && $("consent-name").value.trim().length >= 2;
}

async function giveConsent() {
  const btn = $("consent-btn");
  btn.disabled = true;
  const r = await api("POST", "/api/consent", {
    name: $("consent-name").value,
    age_ok: $("age-ok").checked,
    version: S.consent.version,
  });
  btn.disabled = !consentValid();
  if (!r.ok) return setMsg("consent-msg", r.data.message || "Couldn't save. Try again.", "error");
  route();
}

// ------------------------------------------------------------------ about you

function showAbout() {
  const p = session.profile || {};
  chosenVoice = p.voice_type || chosenVoice;
  $("sex").value = p.sex || "unknown";
  $("device").value = p.device || "";
  $("mic").value = p.mic || "";
  $("tradition").value = p.tradition || "";
  renderVoices();
  $("about-msg").hidden = true;
  show("about");
}

function renderVoices() {
  const locked = session.profile && acceptedCount() > 0;
  $("voice-locked").hidden = !locked;
  const box = $("voices");
  box.textContent = "";
  for (const v of VOICE_ORDER) {
    const a = S.arpeggios[v];
    const card = document.createElement("div");
    card.className = "voice" + (chosenVoice === v ? " selected" : "");
    const pick = document.createElement("button");
    pick.type = "button";
    pick.className = "voice-pick";
    pick.setAttribute("aria-pressed", String(chosenVoice === v));
    pick.disabled = locked && v !== session.profile.voice_type;
    const nm = document.createElement("span");
    nm.className = "voice-name";
    nm.textContent = VOICE_LABEL[v];
    const rg = document.createElement("span");
    rg.className = "voice-range";
    rg.textContent = `${a.notes.slice(0, 4).join(" ")} … (top note ${a.notes[3]})`;
    pick.append(nm, rg);
    pick.addEventListener("click", () => {
      chosenVoice = v;
      renderVoices();
    });
    const play = document.createElement("button");
    play.type = "button";
    play.className = "btn secondary";
    play.textContent = "▶";
    play.setAttribute("aria-label", `Play the ${v} example`);
    play.addEventListener("click", () => playExample(v));
    card.append(pick, play);
    box.append(card);
  }
}

function playExample(voice) {
  if (Rec.busy()) return;
  if (exampleAudio) exampleAudio.pause();
  exampleAudio = new Audio(`/audio/${voice}.wav`);
  exampleAudio.play().catch(() => {});
}

async function saveProfile() {
  const body = {
    voice_type: chosenVoice,
    sex: $("sex").value,
    device: $("device").value,
    mic: $("mic").value,
    tradition: $("tradition").value,
  };
  if (!body.voice_type) return setMsg("about-msg", "Choose a voice type.", "error");
  if (!body.device || !body.mic) return setMsg("about-msg", "Choose your device and microphone.", "error");
  const r = await api("POST", "/api/profile", body);
  if (!r.ok) return setMsg("about-msg", r.data.message || "Couldn't save. Try again.", "error");
  route();
}

// ------------------------------------------------------------------ recorder (spec 5.1 / 5.2)

const Rec = {
  ctx: null,
  stream: null,
  track: null,
  node: null,
  state: "idle", // idle | countin | recording
  chunks: [],
  n: 0,
  timers: [],
  sources: [],
  ui: null, // {phase(text, kind), level(rms)}
  onDone: null,
  heardExample: false,
  exampleBufs: {},
  pre: null, // pre-roll stats while recording

  busy() {
    return this.state !== "idle";
  },

  async init() {
    if (this.ctx && this.track && this.track.readyState === "live") {
      if (this.ctx.state === "suspended") await this.ctx.resume();
      return;
    }
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const e = new Error("insecure");
      e.name = "InsecureError";
      throw e;
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
      video: false,
    });
    this.track = this.stream.getAudioTracks()[0];
    const rate = (this.track.getSettings() || {}).sampleRate;
    const AC = window.AudioContext || window.webkitAudioContext;
    try {
      this.ctx = rate ? new AC({ sampleRate: rate }) : new AC(); // native rate, never resampled
    } catch (e) {
      this.ctx = new AC();
    }
    await this.ctx.audioWorklet.addModule("/static/recorder-worklet.js");
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, "capture", { numberOfInputs: 1, numberOfOutputs: 1 });
    const mute = this.ctx.createGain();
    mute.gain.value = 0;
    src.connect(this.node);
    this.node.connect(mute).connect(this.ctx.destination);
    this.node.port.onmessage = (e) => this._block(e.data);
    this.track.addEventListener("ended", () => {
      this.cancel();
      this.ctx = null;
    });
    if (this.ctx.state === "suspended") await this.ctx.resume();
  },

  settings() {
    const s = this.track ? this.track.getSettings() : {};
    return JSON.parse(JSON.stringify(s || {}));
  },

  async _exampleBuf(voice) {
    if (!this.exampleBufs[voice]) {
      const r = await fetch(`/audio/${voice}.wav`);
      this.exampleBufs[voice] = await this.ctx.decodeAudioData(await r.arrayBuffer());
    }
    return this.exampleBufs[voice];
  },

  _tone(freq, t, dur) {
    const c = this.ctx;
    const osc = c.createOscillator();
    const real = new Float32Array([0, 1, 0.5, 0.3, 0.18, 0.1]);
    osc.setPeriodicWave(c.createPeriodicWave(real, new Float32Array(real.length)));
    osc.frequency.value = freq;
    const g = c.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.22, t + 0.03);
    g.gain.setValueAtTime(0.22, t + dur - 0.08);
    g.gain.linearRampToValueAtTime(0, t + dur);
    osc.connect(g).connect(c.destination);
    osc.start(t);
    osc.stop(t + dur + 0.02);
    this.sources.push(osc);
  },

  _click(t) {
    const c = this.ctx;
    const osc = c.createOscillator();
    osc.frequency.value = 1500;
    const g = c.createGain();
    g.gain.setValueAtTime(0.35, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + 0.05);
    osc.connect(g).connect(c.destination);
    osc.start(t);
    osc.stop(t + 0.06);
    this.sources.push(osc);
  },

  _at(t, fn) {
    this.timers.push(setTimeout(fn, Math.max(0, (t - this.ctx.currentTime) * 1000)));
  },

  // Record: [example the first time] -> start note 1 s -> 3 clicks -> 200 ms -> capture.
  async start(voice, ui, onDone) {
    this.cancel();
    this.ui = ui;
    this.onDone = onDone;
    this.state = "countin";
    const F = S.flow;
    const beat = 60 / F.bpm;
    let t = this.ctx.currentTime + 0.15;
    if (!this.heardExample) {
      ui.phase("Listen to the example…", "wait");
      let buf = null;
      try {
        buf = await this._exampleBuf(voice);
      } catch (e) {
        /* play on without it */
      }
      if (this.state !== "countin") return;
      if (buf) {
        t = this.ctx.currentTime + 0.15;
        const src = this.ctx.createBufferSource();
        src.buffer = buf;
        src.connect(this.ctx.destination);
        src.start(t);
        this.sources.push(src);
        t += buf.duration + 0.8;
      }
      this.heardExample = true;
    }
    const tNote = t;
    this._at(tNote, () => ui.phase("Starting note…", "wait"));
    this._tone(S.arpeggios[voice].hz[0], tNote, F.start_note_s);
    for (let k = 0; k < F.count_in_beats; k++) {
      const tc = tNote + F.start_note_s + k * beat;
      this._click(tc);
      this._at(tc, () => ui.phase(String(F.count_in_beats - k), "wait"));
    }
    const tEnd = tNote + F.start_note_s + (F.count_in_beats - 1) * beat + 0.06;
    this._at(tEnd + F.gap_s, () => this._begin());
  },

  _begin() {
    if (this.state !== "countin") return;
    this.state = "recording";
    this.chunks = [];
    this.n = 0;
    this.pre = { ss: 0, blocks: 0, floor: null, thresh: null, voiced: 0, silent: 0, seen: false };
    this.ui.phase("Wait…", "wait");
    if (this.ui.recording) this.ui.recording();
  },

  _block({ samples, rms }) {
    if (this.ui) this.ui.level(rms);
    if (this.state !== "recording") return;
    const sr = this.ctx.sampleRate;
    const F = S.flow;
    const dt = samples.length / sr;
    this.chunks.push(samples);
    this.n += samples.length;
    const elapsed = this.n / sr;
    const p = this.pre;
    if (p.floor === null) {
      p.ss += rms * rms;
      p.blocks += 1;
      if (elapsed >= F.preroll_s) {
        p.floor = Math.sqrt(p.ss / p.blocks);
        p.thresh = Math.max(0.002, p.floor * 3.16); // 10 dB above the room
        this.ui.phase("Sing now", "sing");
      }
    } else if (rms > p.thresh) {
      p.voiced += dt;
      p.silent = 0;
      if (p.voiced >= 0.15) p.seen = true;
    } else {
      p.voiced = 0;
      if (p.seen) p.silent += dt;
      if (p.seen && p.silent >= F.silence_stop_s) return this.stop();
    }
    if (elapsed >= F.max_take_s) this.stop();
  },

  // Stop tapped, 1.5 s of silence after singing, or the 15 s cap.
  stop() {
    if (this.state !== "recording") return;
    const sr = this.ctx.sampleRate;
    const chunks = this.chunks;
    const n = this.n;
    this.cancel();
    if (n / sr < S.flow.preroll_s + 0.5) {
      this.ui.phase("Stopped before you sang. Tap Record to try again.", "wait");
      if (this.onDone) this.onDone(null);
      return;
    }
    const x = new Float32Array(n);
    let o = 0;
    for (const c of chunks) {
      x.set(c, o);
      o += c.length;
    }
    if (this.onDone) this.onDone(encodeWav(x, Math.round(sr)), n / sr);
  },

  // Restart and close: drop everything in progress. Nothing is uploaded.
  cancel() {
    this.timers.forEach(clearTimeout);
    this.timers = [];
    for (const s of this.sources) {
      try {
        s.stop();
      } catch (e) {
        /* already stopped */
      }
    }
    this.sources = [];
    this.state = "idle";
    this.chunks = [];
    this.n = 0;
  },
};

function encodeWav(f32, sr) {
  const n = f32.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const str = (o, s) => [...s].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)));
  str(0, "RIFF");
  v.setUint32(4, 36 + n * 2, true);
  str(8, "WAVE");
  str(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, 1, true); // mono
  v.setUint32(24, sr, true);
  v.setUint32(28, sr * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, "data");
  v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    v.setInt16(44 + 2 * i, Math.round(s * 32767), true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

// The recorder box: phase text + level meter, inside the mic screen or the slot panel.
function recorderUI(name) {
  const box = document.querySelector(`[data-recorder="${name}"]`);
  box.textContent = "";
  const phase = document.createElement("p");
  phase.className = "rec-phase";
  phase.setAttribute("aria-live", "polite");
  const hint = document.createElement("p");
  hint.className = "rec-hint";
  const meter = document.createElement("div");
  meter.className = "meter";
  meter.setAttribute("aria-hidden", "true");
  const fill = document.createElement("div");
  meter.append(fill);
  const label = document.createElement("p");
  label.className = "meter-label";
  label.textContent = "Mic level";
  box.append(phase, hint, meter, label);
  let last = 0;
  return {
    box,
    phase(text, kind) {
      phase.textContent = text;
      phase.className = "rec-phase" + (kind ? " " + kind : "");
    },
    hint(text) {
      hint.textContent = text || "";
    },
    level(rms) {
      const now = performance.now();
      if (now - last < 50) return;
      last = now;
      const db = 20 * Math.log10(rms + 1e-9);
      fill.style.width = Math.max(0, Math.min(100, ((db + 60) / 60) * 100)) + "%";
      meter.classList.toggle("hot", rms > 0.5);
    },
  };
}

async function ensureMic() {
  try {
    await Rec.init();
    return true;
  } catch (e) {
    showMicHelp(e);
    return false;
  }
}

function showMicHelp(err) {
  const ua = navigator.userAgent;
  const iOS = /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
  const box = $("mic-help-text");
  box.textContent = "";
  const p = (t) => {
    const el = document.createElement("p");
    el.textContent = t;
    box.append(el);
  };
  if (err && err.name === "InsecureError") {
    $("mic-help-title").textContent = "Microphone not available";
    p("This page has to be opened over https:// to use the microphone. Open the link from your invite email exactly as sent.");
  } else if (err && (err.name === "NotFoundError" || err.name === "OverconstrainedError")) {
    $("mic-help-title").textContent = "No microphone found";
    p("Plug in or turn on a microphone (built-in, wired headset or USB), then try again.");
  } else if (err && err.name !== "NotAllowedError" && err.name !== "SecurityError") {
    $("mic-help-title").textContent = "Microphone problem";
    p("The microphone couldn't start. Close other apps that might be using it (calls, voice memos), reload this page and try again.");
  } else {
    $("mic-help-title").textContent = "Microphone is blocked";
    p("This site needs your microphone to record your takes. To turn it back on:");
    if (iOS && /CriOS/.test(ua)) p("Chrome on iPhone/iPad: open the Settings app → Chrome → turn Microphone on. Then reload this page.");
    else if (iOS) p("Safari on iPhone/iPad: tap ᴀA in the address bar → Website Settings → Microphone → Allow. Or open the Settings app → Apps → Safari → Microphone → Allow. Then reload this page.");
    else if (/Android/.test(ua)) p("Chrome on Android: tap the icon to the left of the web address → Permissions → Microphone → Allow. If it's still blocked, open Settings → Apps → Chrome → Permissions → Microphone → Allow. Then reload this page.");
    else if (/Edg\//.test(ua)) p("Edge: click the lock icon to the left of the web address → Permissions for this site → Microphone → Allow. Then reload this page.");
    else if (/Firefox\//.test(ua)) p("Firefox: click the crossed-out microphone icon in the address bar and remove the block. Then reload this page.");
    else if (/Chrome\//.test(ua)) p("Chrome: click the icon to the left of the web address → Site settings → Microphone → Allow. Then reload this page.");
    else if (/Safari\//.test(ua)) p("Safari on Mac: Safari menu → Settings → Websites → Microphone → set this site to Allow. Then reload this page.");
    else p("Open your browser's site settings for this page, allow the microphone, then reload.");
    if (!iOS && !/Android/.test(ua)) p("Still blocked? Check your computer's privacy settings (Mac: System Settings → Privacy & Security → Microphone; Windows: Settings → Privacy → Microphone) and allow your browser.");
  }
  $("mic-help").hidden = false;
}

function uploadForm(blob) {
  const fd = new FormData();
  fd.append("audio", blob, "take.wav");
  fd.append("track_settings", JSON.stringify(Rec.settings()));
  return fd;
}

// ------------------------------------------------------------------ mic check

let micUI = null;

function showMic() {
  micUI = recorderUI("mic");
  micResult = null;
  $("mic-msg").hidden = true;
  if (micReady) micUI.phase("Ready", "");
  else micUI.phase("Microphone off", "wait");
  renderMicBar();
  show("mic");
}

function renderMicBar() {
  const bar = $("mic-bar");
  bar.textContent = "";
  if (!micReady) {
    micUI.hint("Your browser will ask to use the microphone. Tap Allow.");
    bar.append(button("Turn on microphone", "primary big", async () => {
      if (await ensureMic()) {
        micReady = true;
        micUI.phase("Ready", "");
        micUI.hint("Make some sound: the bar should move.");
        Rec.ui = micUI;
        renderMicBar();
      }
    }));
    return;
  }
  if (Rec.busy()) {
    bar.append(
      button("Stop", "primary big", () => Rec.stop(), Rec.state !== "recording"),
      button("Restart", "secondary big", () => startMicTake()),
    );
    return;
  }
  if (micResult && micResult.ok) {
    bar.append(
      button("Try again", "secondary big", () => startMicTake()),
      button("Go to my takes", "primary big", () => route()),
    );
    return;
  }
  bar.append(button("Record", "primary big", () => startMicTake()));
}

async function startMicTake() {
  if (!(await ensureMic())) return;
  $("mic-msg").hidden = true;
  micResult = null;
  micUI.hint("");
  Rec.start(session.profile.voice_type, Object.assign(micUI, { recording: renderMicBar }), async (blob) => {
    renderMicBar();
    if (!blob) return;
    micUI.phase("Checking…", "wait");
    const r = await api("POST", "/api/mic-check", uploadForm(blob));
    if (r.status === 403) return;
    if (!r.ok) {
      micUI.phase("", "");
      setMsg("mic-msg", r.data.message || "Upload failed. Try again.", "error");
    } else {
      micResult = r.data;
      micUI.phase(r.data.ok ? "Mic check passed ✓" : "Let's fix one thing", "");
      const extra = (r.data.notes || []).join(" ");
      if (r.data.ok) setMsg("mic-msg", "Your room and mic sound good. " + extra, "ok");
      else setMsg("mic-msg", r.data.message, "error");
    }
    renderMicBar();
  });
  renderMicBar();
}

// ------------------------------------------------------------------ board

function nextSlot() {
  const s = S.slots.find((x) => session.slots[x.slot].status !== "accepted");
  return s ? s.slot : null;
}

function showBoard() {
  renderBoard();
  show("board");
}

function renderBoard() {
  updateProgress();
  const box = $("slots");
  box.textContent = "";
  for (const cond of ["comfortable", "edge", "pushed"]) {
    const group = S.slots.filter((x) => x.condition === cond);
    const sec = document.createElement("section");
    sec.className = "cond";
    const h = document.createElement("h2");
    h.textContent = `${COND_LABEL[cond]} · takes ${group[0].slot}–${group[group.length - 1].slot}`;
    sec.append(h);
    if (cond === "pushed") {
      const w = document.createElement("p");
      w.className = "warning";
      w.textContent = group[0].warning;
      sec.append(w);
    }
    const ul = document.createElement("ul");
    ul.className = "slots";
    for (const info of group) {
      const st = session.slots[info.slot];
      const status = checkingSlot === info.slot ? "checking" : st.status;
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button";
      b.className = "slot " + status;
      const num = document.createElement("span");
      num.className = "slot-num";
      num.textContent = info.slot;
      const txt = document.createElement("span");
      txt.className = "slot-text";
      const ins = document.createElement("span");
      ins.className = "slot-ins";
      ins.textContent = info.instruction;
      txt.append(ins);
      if (status === "rejected" && st.last_reject) {
        const why = document.createElement("span");
        why.className = "slot-reason";
        why.textContent = st.last_reject.message || "Not accepted. Try again.";
        txt.append(why);
      }
      const chip = document.createElement("span");
      chip.className = "chip " + status;
      chip.textContent = { empty: "Empty", checking: "Checking…", accepted: "✓ Accepted", rejected: "Rejected" }[status];
      b.append(num, txt, chip);
      b.setAttribute("aria-label", `Take ${info.slot}: ${info.instruction} ${chip.textContent}`);
      b.addEventListener("click", () => openPanel(info.slot));
      li.append(b);
      ul.append(li);
    }
    sec.append(ul);
    box.append(sec);
  }
  const bar = $("board-bar");
  bar.textContent = "";
  const next = nextSlot();
  if (next === null) bar.append(button("Submit my 16 takes", "primary big", () => doSubmit()));
  else bar.append(button(`Record take ${next}`, "primary big", () => openPanel(next)));
}

async function doSubmit() {
  const yes = await confirmBox(
    "Submit your takes?",
    "Once you submit, your takes are locked and can't be changed. You'll get your completion code for the payment form.",
    "Submit",
  );
  if (!yes) return;
  const r = await api("POST", "/api/submit");
  if (r.status === 403) return;
  if (!r.ok) return alertBox("Couldn't submit", r.data.message || "Try again.");
  showDone(r.data.completion_code, r.data.payment_form_url);
}

function showDone(code, url) {
  updateProgress();
  $("code").textContent = code;
  const link = $("pay-link");
  if (url) {
    link.href = url;
    link.hidden = false;
    $("pay-text").textContent = "Enter this code in the payment form to get your $10. Keep this page or write the code down.";
  } else {
    link.hidden = true;
    $("pay-text").textContent = "Keep this code. Prem will send you the payment form.";
  }
  show("done");
}

// ------------------------------------------------------------------ slot panel

let slotUI = null;

function openPanel(slot) {
  openSlot = slot;
  slotResult = null;
  slotUI = recorderUI("slot");
  slotUI.box.hidden = true;
  $("panel").hidden = false;
  document.body.style.overflow = "hidden";
  renderPanel();
  loadPlayback();
}

function closePanel(force) {
  if (!force && checkingSlot !== null && checkingSlot === openSlot) return;
  Rec.cancel();
  if (micUI) Rec.ui = micUI;
  $("panel").hidden = true;
  document.body.style.overflow = "";
  const a = $("panel-audio");
  a.pause();
  a.removeAttribute("src");
  if (playbackUrl) URL.revokeObjectURL(playbackUrl);
  playbackUrl = null;
  openSlot = null;
  if (session && !document.querySelector('[data-screen="board"]').hidden) renderBoard();
}

async function loadPlayback() {
  const slot = openSlot;
  const a = $("panel-audio");
  if (playbackUrl) URL.revokeObjectURL(playbackUrl);
  playbackUrl = null;
  a.removeAttribute("src");
  $("panel-playback").hidden = true;
  if (session.slots[slot].status !== "accepted") return;
  let r;
  try {
    r = await fetch(`/api/takes/${slot}/audio`, { headers: { "X-Invite-Token": TOKEN } });
  } catch (e) {
    return;
  }
  if (!r.ok || openSlot !== slot) return;
  playbackUrl = URL.createObjectURL(await r.blob());
  a.src = playbackUrl;
  $("panel-playback").hidden = Rec.busy();
}

function renderPanel() {
  const slot = openSlot;
  const info = S.slots[slot - 1];
  const st = session.slots[slot];
  $("panel-title").textContent = `Take ${slot} · ${COND_LABEL[info.condition]}${info.loud ? ", loud" : ""}`;
  $("panel-instruction").textContent = info.instruction;
  $("panel-warning").hidden = !info.warning;
  $("panel-warning").textContent = info.warning || "";

  const status = $("panel-status");
  const busy = Rec.busy();
  const checking = checkingSlot === slot;
  if (checking) {
    status.className = "status-line";
    status.textContent = "Checking your take…";
  } else if (slotResult && !slotResult.accepted) {
    status.className = "status-line bad";
    status.textContent = "Not accepted: " + (slotResult.message || "try again.") + (st.status === "accepted" ? " Your earlier take is kept." : "");
  } else if (slotResult && slotResult.accepted) {
    status.className = "status-line good";
    status.textContent = "✓ Accepted and saved. " + (slotResult.notes || []).join(" ");
  } else if (st.status === "accepted") {
    status.className = "status-line good";
    status.textContent = "✓ Accepted. Play it back, record it again, or delete it.";
  } else if (st.status === "rejected" && st.last_reject) {
    status.className = "status-line bad";
    status.textContent = "Last try wasn't accepted: " + (st.last_reject.message || "try again.");
  } else {
    status.className = "status-line";
    status.textContent = "Not recorded yet.";
  }
  $("panel-playback").hidden = busy || checking || st.status !== "accepted" || !playbackUrl;
  slotUI.box.hidden = !(busy || checking);

  const bar = $("panel-bar");
  bar.textContent = "";
  if (busy) {
    bar.append(
      button("Stop", "primary big", () => Rec.stop(), Rec.state !== "recording"),
      button("Restart", "secondary big", () => startSlotTake()),
    );
    return;
  }
  if (checking) {
    bar.append(button("Checking…", "primary big", null, true));
    return;
  }
  const next = nextSlot();
  if (st.status === "accepted") {
    if (slotResult && slotResult.accepted && next !== null && next !== slot) {
      bar.append(button(`Next: take ${next}`, "primary big", () => {
        closePanel();
        openPanel(next);
      }));
    }
    bar.append(
      button("Re-record", "secondary big", () => startSlotTake()),
      button("Delete", "danger big", () => deleteSlot()),
    );
    if (!(slotResult && slotResult.accepted && next !== null && next !== slot)) {
      bar.append(button("Back to takes", "primary big", () => closePanel()));
    }
  } else {
    bar.append(
      button("▶ Hear example", "secondary big", () => playExample(session.profile.voice_type)),
      button(st.status === "rejected" || slotResult ? "Record again" : "Record", "primary big", () => startSlotTake()),
    );
  }
}

async function startSlotTake() {
  if (!(await ensureMic())) return;
  if (exampleAudio) exampleAudio.pause();
  $("panel-audio").pause();
  slotResult = null;
  const slot = openSlot;
  slotUI.hint(S.slots[slot - 1].instruction);
  Rec.start(session.profile.voice_type, Object.assign(slotUI, { recording: renderPanel }), async (blob) => {
    if (!blob) return renderPanel();
    checkingSlot = slot;
    slotUI.phase("Checking…", "wait");
    renderPanel();
    const r = await api("POST", `/api/takes/${slot}`, uploadForm(blob));
    checkingSlot = null;
    if (r.status === 403) return;
    if (r.ok) slotResult = r.data;
    else slotResult = { accepted: false, message: r.data.message || "Upload failed. Try again." };
    updateProgress();
    if (openSlot === slot) {
      renderPanel();
      if (slotResult.accepted) loadPlayback();
    }
  });
  renderPanel();
}

async function deleteSlot() {
  const slot = openSlot;
  const yes = await confirmBox(`Delete take ${slot}?`, "The recording is deleted and the slot goes back to empty.", "Delete");
  if (!yes) return;
  const r = await api("DELETE", `/api/takes/${slot}`);
  if (r.status === 403) return;
  if (!r.ok) return alertBox("Couldn't delete", r.data.message || "Try again.");
  slotResult = null;
  if (openSlot === slot) {
    await loadPlayback();
    renderPanel();
  }
  updateProgress();
}

// ------------------------------------------------------------------ helpers

function button(label, cls, onClick, disabled) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "btn " + cls;
  b.textContent = label;
  if (disabled) b.disabled = true;
  if (onClick) b.addEventListener("click", onClick);
  return b;
}

function setMsg(id, text, kind) {
  const el = $(id);
  el.textContent = text;
  el.className = "msg" + (kind ? " " + kind : "");
  el.hidden = !text;
}

function confirmBox(title, text, yesLabel) {
  return new Promise((resolve) => {
    $("confirm-title").textContent = title;
    $("confirm-text").textContent = text;
    const yes = $("confirm-yes");
    const no = $("confirm-no");
    yes.textContent = yesLabel || "OK";
    no.hidden = false;
    $("confirm").hidden = false;
    yes.focus();
    const done = (v) => {
      $("confirm").hidden = true;
      yes.onclick = no.onclick = null;
      resolve(v);
    };
    yes.onclick = () => done(true);
    no.onclick = () => done(false);
  });
}

function alertBox(title, text) {
  confirmBox(title, text, "OK");
  $("confirm-no").hidden = true;
}

// ------------------------------------------------------------------ wiring

document.addEventListener("click", (e) => {
  const el = e.target.closest("[data-action]");
  if (!el) return;
  const a = el.dataset.action;
  if (a === "go-consent") show("consent");
  else if (a === "give-consent") giveConsent();
  else if (a === "save-profile") saveProfile();
  else if (a === "hear-example") playExample(session.profile.voice_type);
  else if (a === "edit-profile") showAbout();
  else if (a === "redo-mic") showMic();
  else if (a === "close-panel") closePanel();
  else if (a === "close-mic-help") $("mic-help").hidden = true;
  else if (a === "copy-code") {
    const code = $("code").textContent;
    (navigator.clipboard ? navigator.clipboard.writeText(code) : Promise.reject())
      .then(() => ($("copy-btn").textContent = "Copied ✓"))
      .catch(() => ($("copy-btn").textContent = "Select the code above to copy it"));
  }
});

["age-ok", "consent-name"].forEach((id) =>
  $(id).addEventListener("input", () => ($("consent-btn").disabled = !consentValid())),
);
$("age-ok").addEventListener("change", () => ($("consent-btn").disabled = !consentValid()));

document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!$("confirm").hidden) $("confirm-no").click();
  else if (!$("mic-help").hidden) $("mic-help").hidden = true;
  else if (!$("panel").hidden) closePanel();
});

load();
