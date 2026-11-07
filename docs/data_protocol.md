# StrainlessAI: recording protocol for singers

Consent version `consent-v1` · 2026-11-07 · Source: `docs/SPEC_v2.md` Part 2

## What this is

StrainlessAI listens to one short sung exercise and says whether it sounds **healthy** or **strained**. It is an independent project by Prem Nadesan, a Purdue student. It is not a Purdue-sponsored study. To build it, we need many singers recording the same exercise, sometimes easily and sometimes with extra effort. Two raters (Prem and a voice teacher) listen to every take without knowing who sang it or what they were asked to do, and rate how much strain they hear. Those ratings teach the model.

You record **16 short takes** on your own phone or laptop through a private link. It takes about **20–30 minutes**, and you can stop and come back later. You're paid **$10** once all 16 takes are submitted.

## Before you start

- You must be **18 or older**.
- Find a quiet room: no fan, TV or music. Headphones aren't needed.
- Use your device's built-in mic, a wired headset or a USB mic. **Don't use Bluetooth earbuds**; they cut the sound quality and takes will be rejected.
- Keep the mic about an arm's length away, and don't change the distance between takes.
- Warm up as you normally would and keep water nearby.
- The site starts with a short mic check and tells you what to fix (move back, find a quieter room, switch mics) before any real take.

## The exercise

Sing a major arpeggio **1-3-5-8-5-3-1** on **"ah"**. Each note gets one beat, except the **top note, which you hold for 2 beats**. The example plays at about **90 BPM**, which is roughly 5 seconds of singing.

- Hold each note **plain**: no vibrato added on purpose, no slides between notes, no ornaments. Copy the example.
- Exact tempo doesn't matter. Just give each note a clear, steady moment. If you go too fast, the site says "Slow down a little and hold each note."
- Each take: tap **Record**, hear the starting note and 3 clicks, wait while the screen says **"Wait…"** (one second of silence to measure the room), then sing when it says **"Sing now"**. Recording stops when you tap Stop or go quiet for 1.5 seconds. **Restart** throws away a take in progress.

Pick the arpeggio for your voice type. Basses use the baritone arpeggio.

| Voice type | Notes | Top note |
|---|---|---|
| Baritone (and bass) | F3 A3 C4 **F4** C4 A3 F3 | F4, ~349 Hz |
| Tenor | Ab3 C4 Eb4 **Ab4** Eb4 C4 Ab3 | Ab4, ~415 Hz |
| Alto (contralto, mezzo-soprano) | F4 A4 C5 **F5** C5 A4 F4 | F5, ~698 Hz |
| Soprano | Ab4 C5 Eb5 **Ab5** Eb5 C5 Ab4 | Ab5, ~831 Hz |

## Your 16 takes

**The notes never change between takes. Only the effort changes.** Record the takes in this order so the effortful ones come last and can't tire your voice before the easy ones.

| Slots | Condition | What to do |
|---|---|---|
| 1–4 | comfortable | Easy, healthy singing. |
| 5–6 | comfortable, loud | **Loud but free**: a healthy belt, no squeezing. |
| 7–10 | edge | A little more effort than comfortable, but not fully pushed. |
| 11–14 | pushed | Same notes with too much effort: squeezed, pressed, "shouty" throat. |
| 15–16 | pushed, moderate volume | Pushed and squeezed, but **not loud**. |

Slots 5–6 and 15–16 matter: they teach the model that loud isn't the same as strained, and strained isn't always loud.

> **Stop if anything hurts or feels wrong. A few strained takes are enough; never push through pain.**

Each take is checked within a few seconds. An accepted take gets a ✓, and you can play it back, re-record it or delete it. A rejected take isn't kept, and the site tells you why (for example, too noisy, clipped, or a note missing) so you can try again. Your progress is saved after every accepted take. When all 16 are accepted, tap **Submit**. This locks the session, and you get a **completion code** and a link to the payment form.

## Consent (shown on the site, version `consent-v1`)

> **StrainlessAI singing recordings: consent**
>
> You're invited to record 16 short takes of a sung exercise to help build StrainlessAI, a tool that estimates whether singing sounds healthy or strained. StrainlessAI is an independent project by Prem Nadesan, a Purdue student. It is not a Purdue-sponsored study. StrainlessAI is not a medical service, and nothing in this session is a medical assessment of your voice.
>
> **What you'll do.** Sing one short arpeggio 16 times on your own device, taking about 20–30 minutes. Some takes ask you to sing with extra effort. Stop if anything hurts or feels wrong. You never have to push through discomfort. Payment ($10) is made after all 16 takes are submitted.
>
> **What is kept.** Your accepted recordings, the details you enter (voice type, and optionally sex and singing tradition), your device and microphone type, and technical browser audio settings. Recordings are labeled with a random ID, never your name. They are stored in a private cloud bucket, listened to by the project's raters, and used to train and test the StrainlessAI model. They are not made public. Your typed name and this consent are stored separately from your recordings. This site does not store your email address or payment details.
>
> **Your choices.** Taking part is voluntary, and you can stop at any time. You can withdraw later by replying to your invite email: your recordings and consent record will be deleted, and they will be left out of every future version of the model.
>
> By ticking the box and typing your full name, you confirm that you are 18 or older, that you have read this, and that you agree to take part.

The site stores the consent record (singer_id, typed name, consent version, timestamp) under `consent/` in the private bucket, separate from the recordings. Any change to the text above needs a new version (`consent-v2`, ...).

## Your data

- **Recordings never carry your name.** Your name exists only in the consent record.
- **Your email** and its link to your singer ID live only in Prem's private recruiting sheet, never on the site.
- **Your payment handle** (Venmo or Zelle) lives only in the payment Google Form and is deleted once you're paid.
- **Withdrawing** deletes your recordings, consent record and invite from the bucket and from Prem's machine. You're left out of the next training run.

## What's recorded with each take

Every accepted take is one mono WAV file (16-bit PCM, your device's own sample rate) plus one JSON file with these fields. **You** = you choose it. **Site** = filled in automatically.

| Field | Set by | Meaning | Example |
|---|---|---|---|
| `clip_id` | site | Random ID for this take | `c0042` |
| `singer_id` | site | Random ID for you, from your invite | `s007` |
| `session_id` | site | Your recording session | `s007-1` |
| `recorded_at` | site | Date and time, ISO 8601 | `2026-11-20T14:05:00-05:00` |
| `voice_type` | you | `tenor`, `baritone`, `alto` or `soprano` | `alto` |
| `arpeggio_id` | site | Voice type + lowest note: `baritone_F3`, `tenor_Ab3`, `alto_F4`, `soprano_Ab4` | `alto_F4` |
| `condition` | site | `comfortable`, `edge` or `pushed`, from the slot | `pushed` |
| `slot` | site | 1–16 | `12` |
| `loud` | site | `true` if you were asked to sing loudly (slots 5–6 only) | `false` |
| `sex` | you, optional | `F`, `M` or `unknown` (`unknown` if left blank) | `unknown` |
| `tradition` | you, optional | Free text. Kept for later work, not used in v2 | `Carnatic` |
| `device` | you | Device (`phone`, `laptop`, `tablet`, `desktop`) and mic (`built-in`, `wired headset`, `USB mic`, `other`) | `phone / built-in` |
| `user_agent` | site | Your browser's ID string | browser UA string |
| `track_settings` | site | The mic's technical settings from the browser | `track.getSettings()` output |
| `sample_rate` | site | Samples per second | `48000` |
| `duration_s` | site | Length of the take in seconds | `9.8` |
| `finder_low`, `finder_high` | site, optional | Your lowest and highest notes, if you used the voice-type finder | `E3`, `A5` |

No name, email, payment details or IP address are ever stored with a take.
