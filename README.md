# jevbot

Discord bot that makes [TypeSafe's Jev](https://openrouter.ai/~typesafe/jev-latest) talk — a decision model that "cannot generate text," loomed word-by-word into broken sentences.

Jev is a non-autoregressive decision model. It answers questions with calibrated probabilities, not text. This bot gives it a 20K word vocabulary and asks "next word?" repeatedly via tournament sampling until it forms a reply.

A fork of [lyramakesmusic/jevbot](https://github.com/lyramakesmusic/jevbot), which runs against hosted Jev only. This fork adds a self-hosted [Laya](https://laya.convaiinnovations.com/) backend, packaging for macOS and Linux, and CI publishing images to `ghcr.io/cschmittiey/jevbot` and `ghcr.io/cschmittiey/jevbot-laya`.

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

Through a local Laya, for comparison:

- "Unlimited great admire praise admiration jazzy"
- "Answers phones answered information users info"

## Setup

```bash
pip install -r requirements.txt
```

`.env` needs `DISCORD_TOKEN_JEV=...`, plus `OPENROUTER_API_KEY=...` for hosted Jev, or `JEV_BACKEND=laya` with `LAYA_URL` and `LAYA_API_KEY` for Laya. Enable **Message Content Intent** in the Discord developer portal.

```bash
python jev_bot.py            # hosted Jev
python jev_bot.py --laya     # self-hosted Laya
```

## Running Laya

```bash
python3 -m venv .venv && .venv/bin/pip install "laya[serve]==0.3.20"
LAYA_DEVICE=mps LAYA_PRELOAD=1 LAYA_MODELS=english LAYA_API_KEY=$(openssl rand -hex 32) \
  .venv/bin/laya-serve
```

`LAYA_DEVICE` is `cpu`, `cuda` or `mps`; on Apple silicon `mps` uses the GPU but needs a native process, since a container cannot see it. Laya's option and question budgets are much smaller than Jev's, so this backend scores 200 candidates per word step instead of the whole 20K vocab — the knobs and the measured limits are commented in `jev_bot.py`.

## Deploying

Both images are on ghcr, so deploying is pull and run:

```bash
cp deploy/jevbot.env.example deploy/jevbot.env   # Discord token + Laya key
cp deploy/laya.env.example  deploy/laya.env      # the same Laya key
docker compose pull && docker compose up
```

`docker compose up --build` builds from source instead. The bot image is distroless (127MB, no torch); Laya is separate because it carries torch and the weights. NVIDIA hosts build with `LAYA_TORCH_INDEX=cu128`.

`deploy/` also has systemd units for each half and a launchd plist that runs Laya natively on macOS for Apple GPU inference — the fastest setup on a Mac, with the bot in a container and `LAYA_URL=http://host.docker.internal:8000`. The units and the plist **have not been executed**; only the Docker path has been built and run.

## Usage

Mention jev or reply to jev's messages. Replies only — it won't respond to messages that don't involve it.

## Cost

Against hosted Jev, ~$0.01-0.05 per reply via OpenRouter at ~6 API calls per word. Against a local Laya it is free, and a word step measures ~0.28s on an M-series GPU.

## Vocab

`vocab.txt` is a 20K word list (from [bewinxed/jevgpt](https://github.com/bewinxed/jevgpt)) with slurs removed. Words can be added or removed freely — the vocab IS the content filter.

## Credits

- [TypeSafe AI](https://typesafe.ai) for Jev
- [bewinxed/jevgpt](https://github.com/bewinxed/jevgpt) for the tournament sampling architecture and vocab
- [ConvAI Innovations](https://laya.convaiinnovations.com/) for Laya, the Apache-2.0 decision engine
- Built by [lyra](https://twitter.com/_lyraaaa_) + clod; Laya backend and packaging by [cschmittiey](https://github.com/cschmittiey)
