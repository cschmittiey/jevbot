# jevbot

Discord bot that makes [TypeSafe's Jev](https://openrouter.ai/~typesafe/jev-latest) talk — a decision model that "cannot generate text," loomed word-by-word into broken sentences.

Jev is a non-autoregressive decision model. It answers questions with calibrated probabilities, not text. This bot gives it a 20K word vocabulary and asks "next word?" repeatedly via tournament sampling until it forms a reply.

It can loom against either backend:

| | `jev` (default) | `laya` |
| --- | --- | --- |
| Where | TypeSafe's hosted API via OpenRouter | self-hosted [`laya-serve`](https://laya.convaiinnovations.com/) |
| Options per question | up to 255 | ~150 bare words — a token budget, not a count |
| Questions per request | no local limit | **64 hard cap** (`413` above it, and the noul counts) |
| Candidates per word step | the whole 20K vocab | 200 by default (20 questions x 10 words) |
| Cost per reply | ~$0.01–0.05 | $0 — your own hardware |
| Word-step latency | network round trips | ~0.28s measured on an M-series GPU |

Laya speaks the same `POST /v1/systemone` protocol and returns the same answer shape (`answers[qid].probabilities`, `answers[qid].noul`), so the swap is the base URL, a bearer key and the request shape. See [Backends](#backends) for the measured limits.

## How it works

1. **Tournament sampling**: the 20K vocab shuffled into buckets, all scored in parallel — 255-word buckets against hosted Jev, 20 x 10 against Laya
2. **Runoff**: Top-2 from each bucket compete in a final round
3. **Completeness judge**: A separate `noul` question asks "is the reply complete?" — jev stops when it thinks it's done
4. **Penalty system**: Content words penalized 2.5x per reuse, stopwords 1.6x — prevents "is are I is are" loops
5. **User-only history**: Last 3 user messages included as context; jev's own broken output is excluded (it poisons follow-ups)

## Output examples

Hosted Jev:

- "I love jazz because its improvised and freedom."
- "Rock.? Yeah"
- "No because overkill. Overkill!.!.!"
- "I depends on on situation of circumstances."
- "Yuck no ugh spit! Gag gagging ing"
- "Band is from california in san los angeles. Las angels."

Real replies loomed through a local Laya, for comparison:

- "Unlimited great admire praise admiration jazzy"
- "Answers phones answered information users info"
- "Share user entering entered scams invaders"
- "Ok verdict okay alright accept accepts ok uncertainty verdict simpler verdict"

## Backends

`JEV_BACKEND=jev` (default) or `JEV_BACKEND=laya`; `--laya` is shorthand for the latter.
The Laya backend needs `LAYA_URL` (default `http://127.0.0.1:8000`) and `LAYA_API_KEY` if the
server was started with one. Both secrets accept a `*_FILE` sibling
(`DISCORD_TOKEN_JEV_FILE`, `LAYA_API_KEY_FILE`) so container secrets need not be env vars.

### Laya's limits, measured

Laya's budgets are smaller than Jev's and two of them bite this bot directly. All measured
against `laya-serve` 0.3.20 with the english checkpoint:

- **Options per question are a token budget, not a count.** A question's options share
  `head_max_len` (192 tokens; 256 on the multilingual/typed-decisions checkpoints) and the
  whole sequence shares `max_len` (512). 150 bare vocab words passed on 8/8 random slices,
  200 on 5/8, 220 on 0/8. Going over is a hard `422`, not a silent trim — and because the
  cost depends on which words were drawn, the ceiling moves from step to step.
- **64 questions per request**, a module constant in `laya/serve.py` and not configurable.
  The completeness `noul` counts, so the ceiling is 63 word questions plus it. Over is a `413`.
- **Choice questions with 11+ options are uncalibrated.** The checkpoint ships
  `choice:11+=0.1006`, below Laya's `TEMP_MIN` of 0.5, so the runtime clamps it to 0.5 and
  warns on load. This bot samples *relatively* from those probabilities rather than
  thresholding on them, so the ranking still works — but the distribution's sharpness is not
  the trained one.
- So this backend scores **200 candidates per word step, not the whole vocab**. That is the
  real cost of self-hosting here. Raise `JEV_QUESTIONS_PER_STEP` (up to 63) or
  `JEV_OPTIONS_PER_QUESTION` (stay under ~150) to trade latency for coverage: 40 x 10
  (~320 candidates) measures ~0.4s per step instead of ~0.28s.

If a request is rejected anyway, the bot halves the bucket size, retries, and remembers the
size that fit for the rest of the process, so it pays that round trip once rather than per step.

## Setup

```bash
pip install -r requirements.txt
```

Create `.env`:
```
DISCORD_TOKEN_JEV=your_discord_bot_token
```

For the hosted backend, add `OPENROUTER_API_KEY=...`. For Laya, add:
```
JEV_BACKEND=laya
LAYA_URL=http://127.0.0.1:8000
LAYA_API_KEY=whatever_you_started_laya_serve_with
```

Enable **Message Content Intent** in Discord developer portal.

## Running Laya

`laya-serve` needs Python 3.10+, torch, and ~800MB of weights on first boot:

```bash
python3 -m venv .venv && .venv/bin/pip install "laya[serve]==0.3.20"
LAYA_DEVICE=mps LAYA_PRELOAD=1 LAYA_MODELS=english LAYA_API_KEY=$(openssl rand -hex 32) \
  .venv/bin/laya-serve          # -> {"status":"ok","loaded":["english"],"device":"mps"}
```

`LAYA_DEVICE` is `cpu`, `cuda` or `mps` — on Apple silicon `mps` uses the GPU and needs a
native process, because a Linux container cannot see the Apple GPU. `LAYA_PRELOAD=1` builds
checkpoints before the server binds, so a healthy `/health` means ready. `LAYA_MODELS=english`
keeps the download to ~800MB instead of the 2.5GB bundle.

```bash
python jev_bot.py            # hosted Jev
python jev_bot.py --laya     # self-hosted Laya
```

## Deploying

Three paths, same bot code.

**Containers (macOS or Linux).** The bot image is distroless (127MB, no torch); Laya is a
separate service because it carries torch and the weights.

```bash
cp deploy/jevbot.env.example deploy/jevbot.env   # Discord token + Laya key
cp deploy/laya.env.example  deploy/laya.env      # the same Laya key
docker compose up --build
```

`laya-serve` publishes on loopback only and preloads before it reports healthy, so `up` waits
for a usable engine. NVIDIA hosts: build with `LAYA_TORCH_INDEX=cu128` and add a GPU
reservation to the `laya-serve` service.

**systemd (Linux, no containers).** `deploy/jevbot.service` and `deploy/laya-serve.service`
run a venv install of each half, with install instructions in the unit comments. The Laya unit
passes its bearer token as a systemd credential rather than an environment file, so the key
never lands in a unit or an env file.

**launchd (macOS, for Apple GPU inference).** `deploy/laya-serve.plist` runs Laya natively with
`LAYA_DEVICE=mps`; the paths inside are Mac-specific and need adjusting.

Mixing is fine, and is the fastest setup on a Mac: Laya native on the GPU, bot in the
container, with `LAYA_URL=http://host.docker.internal:8000` in `deploy/jevbot.env`.

The systemd units and the plist were written on macOS and **have not been executed** — there is
no systemd here, and the launchd job was not loaded. The Docker path was built and run: the
image imports its deps on distroless, fails cleanly without a token, and reaches Laya from
inside the container.

## Usage

Mention jev or reply to jev's messages. Replies only — it won't respond to messages that don't involve it.

## Cost

Against hosted Jev, ~$0.01-0.05 per reply via OpenRouter. Tournament sampling does ~6 API
calls per word. Against a local Laya, inference is free: a word step is two batched forward
passes (a sweep and a runoff), measured at ~0.28s plus ~0.19s for a 20 x 10 step on an
M-series GPU. Laya's response includes a `usage` block (`input_tokens`, always 0 output
tokens — it never generates text) if you want to size hardware.

## Vocab

`vocab.txt` is a 20K word list (from [bewinxed/jevgpt](https://github.com/bewinxed/jevgpt)) with slurs removed. Words can be added or removed freely — the vocab IS the content filter.

## Credits

- [TypeSafe AI](https://typesafe.ai) for Jev
- [bewinxed/jevgpt](https://github.com/bewinxed/jevgpt) for the tournament sampling architecture and vocab
- [ConvAI Innovations](https://laya.convaiinnovations.com/) for Laya, the Apache-2.0 decision engine
- Built by [lyra](https://twitter.com/_lyraaaa_) + clod
