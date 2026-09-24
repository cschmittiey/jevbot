"""
Jev Discord Bot — tournament-sampled loomed text from TypeSafe's decision model.

v1: empty descriptions, "Next word?" — the original broken-grammar jev
v2: mode D descriptions ("...lastword candidate"), better instructions — more coherent

Run with --v1 for the original style, default is v2.

Run with --laya (or JEV_BACKEND=laya) to loom against a self-hosted laya-serve instead of
the hosted decisions API. Laya speaks the same /v1/systemone protocol and returns the same
answer shape, but it enforces much smaller option and question budgets, so each step
samples the vocab in 20x10 buckets rather than scoring all 255-per-question.
"""

import os
import re
import sys
import asyncio
import logging
import random
from pathlib import Path
from collections import defaultdict

import aiohttp
import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()


def _env_or_file(name):
    """Read NAME, or the file named by NAME_FILE -- keeps container secrets out of the env."""
    path = os.environ.get(f"{name}_FILE")
    if path:
        return Path(path).read_text().strip()
    return os.environ.get(name)


TOKEN = _env_or_file("DISCORD_TOKEN_JEV")
if not TOKEN:
    raise SystemExit("DISCORD_TOKEN_JEV (or DISCORD_TOKEN_JEV_FILE) is required")
OPENROUTER_KEY = _env_or_file("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "~typesafe/jev-latest"
END = "<END>"

# Backend: "jev" is the hosted OpenRouter decisions API, "laya" is a self-hosted
# laya-serve. Laya speaks the same /v1/systemone wire protocol, so the swap is the base
# URL, the bearer key and the request shape -- not the response shape.
BACKEND = os.environ.get("JEV_BACKEND", "laya" if "--laya" in sys.argv else "jev")
LAYA_URL = os.environ.get("LAYA_URL", "http://127.0.0.1:8000").rstrip("/")
LAYA_API_URL = LAYA_URL + "/v1/systemone"
LAYA_KEY = _env_or_file("LAYA_API_KEY") or ""

# Measured limits on laya-serve 0.3.20 / english checkpoint. MAX_QUESTIONS=64 is a hard
# module constant in laya/serve.py (413 above it, and the completeness noul counts), and
# the per-question option ceiling is a token budget rather than a count: 150 bare vocab
# words passed on 8/8 random slices, 200 on 5/8, 220 on 0/8. Jev's hosted API takes 255
# options per question, so this is the one place the backends really differ.
LAYA_MAX_QUESTIONS = 63   # 64 minus the "complete" noul
LAYA_MAX_OPTIONS = 150
LAYA_QUESTIONS_PER_STEP = int(os.environ.get("JEV_QUESTIONS_PER_STEP", 20))
LAYA_OPTIONS_PER_QUESTION = int(os.environ.get("JEV_OPTIONS_PER_QUESTION", 10))

# The option ceiling is a token budget that moves with the words drawn, so a request can be
# rejected at a size that worked moments earlier. Once a size is known to fit, remember it,
# or every later step pays for the rejected round trip again.
_laya_options_fit = None


def _laya_option_budget():
    cap = LAYA_MAX_OPTIONS if _laya_options_fit is None else min(LAYA_MAX_OPTIONS, _laya_options_fit)
    return max(2, min(LAYA_OPTIONS_PER_QUESTION, cap))

# Version flag
V1_MODE = "--v1" in sys.argv

MAX_CHOICES = 255
QUESTIONS_PER_CALL = 20 if V1_MODE else 10  # mix mode averages out; pure v1 can pack more
TOP_PER_BUCKET = 2
MAX_WORDS = 30
MIN_WORDS = 2
MAX_HISTORY = 5
STOP_THRESHOLD = 0.5
REPEAT_PENALTY = 1.5
REPEAT_WINDOW = 8
CONTENT_PENALTY = 2.5
CONTENT_PENALTY_CAP = 4
STOP_PENALTY = 1.6
STOP_PENALTY_CAP = 6

STOPWORDS = set(
    "a an the and or but if of to in on at by for with from as is are was were be been "
    "being it its this that these those i you he she they we me him her them us my your "
    "his their our not no so then than there here when where which who what how all any "
    "some each into over under about above below up down out off again more most very "
    "can will just do does did have has had would could should may might must".split()
)
NO_SPACE_BEFORE = set(".,!?;:)\"'")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("jev")

async def fetch_history(channel, before_msg):
    """Grab last N user messages from channel — rebuilds context from Discord, no caching."""
    lines = []
    try:
        async for msg in channel.history(limit=30, before=before_msg):
            if msg.author.bot:
                continue
            text = (msg.content or "").strip()
            if not text:
                continue
            # Skip bot commands
            if text.startswith("."):
                continue
            lines.append(text)
            if len(lines) >= MAX_HISTORY:
                break
    except Exception:
        pass
    lines.reverse()
    return [{"role": "user", "content": t} for t in lines]

# Vocab
VOCAB_PATH = Path(__file__).parent / "vocab.txt"
BANNED = {"unanswered", "\\n"}
BASE_VOCAB = [w for w in VOCAB_PATH.read_text().split("\n") if w and w.lower() not in BANNED]
log.info(f"Loaded {len(BASE_VOCAB)} vocab words | {'v1' if V1_MODE else 'mix'} mode")
if BACKEND == "laya":
    log.info(f"backend: laya -> {LAYA_API_URL} "
             f"({LAYA_QUESTIONS_PER_STEP}x{LAYA_OPTIONS_PER_QUESTION} options per step)")
else:
    if not OPENROUTER_KEY:
        raise SystemExit("OPENROUTER_API_KEY is required for the jev backend "
                         "(set it, or run with --laya / JEV_BACKEND=laya)")
    log.info(f"backend: jev -> {API_URL} ({MODEL})")


def render(tokens):
    out = ""
    for t in tokens:
        if t == "\\n" or t == "\n":
            continue
        if not out or out.endswith(("\n", " ")) or t in NO_SPACE_BEFORE:
            out += t
        else:
            out += " " + t
    return re.sub(r"(^|[.!?]\s+|\n)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out.strip())


def vocabulary(message):
    seen = set(BASE_VOCAB)
    extra = [w for w in re.findall(r"[A-Za-z']+", message.lower())
             if w not in seen and not seen.add(w)]
    return BASE_VOCAB + extra + [END]


def penalty(reply, word):
    local = reply[-REPEAT_WINDOW:].count(word) + 2 * (reply[-1:] == [word])
    p = REPEAT_PENALTY ** local
    seen = reply.count(word)
    if word.isalpha() and word.lower() not in STOPWORDS:
        p *= CONTENT_PENALTY ** min(seen, CONTENT_PENALTY_CAP)
    elif word.isalpha():
        p *= STOP_PENALTY ** min(seen, STOP_PENALTY_CAP)
    return p


async def post(session, state, questions):
    body = {"model": MODEL, "state": state, "questions": questions}
    for attempt in range(3):
        try:
            async with session.post(API_URL, json=body, timeout=aiohttp.ClientTimeout(total=30)) as r:
                data = await r.json()
                if r.status < 400:
                    return data.get("answers", {})
                log.warning(f"API {r.status}: {str(data.get('error',''))[:200]}")
                await asyncio.sleep(1 + 2 * attempt)
        except Exception as e:
            log.warning(f"API err {attempt}: {e}")
            await asyncio.sleep(1 + 2 * attempt)
    return {}


async def laya_post(session, state, questions):
    """POST one laya-serve request. Returns (status, answers, detail); status 0 = transport
    failure and status 413/422 carry the server's detail string.

    Unlike the hosted API, laya's errors are actionable: 413 when a request carries more
    than 64 questions, 422 when one question's options overflow its token budget. Both are
    returned to the caller to shrink and retry instead of being retried blindly.
    """
    body = {"state": state, "questions": questions}
    for attempt in range(3):
        try:
            async with session.post(LAYA_API_URL, json=body,
                                    timeout=aiohttp.ClientTimeout(total=30)) as r:
                if r.status < 400:
                    return r.status, (await r.json()).get("answers", {}), ""
                detail = ""
                try:
                    detail = str((await r.json()).get("detail", ""))[:200]
                except Exception:
                    pass
                if r.status in (413, 422):
                    return r.status, {}, detail
                log.warning(f"laya {r.status}: {detail}")
                await asyncio.sleep(1 + 2 * attempt)
        except Exception as e:
            log.warning(f"laya err {attempt}: {e}")
            await asyncio.sleep(1 + 2 * attempt)
    return 0, {}, "transport failure"


def choice_q(words, reply_so_far="", force_mode=None):
    mode = force_mode or ("v1" if V1_MODE else "mix")
    if mode == "v1":
        return {"type": "choice", "instructions": "Next word?",
                "criteria": {w: "" for w in words}}
    elif mode == "v2":
        last = reply_so_far.split()[-1] if reply_so_far.split() else ""
        criteria = {w: f"...{last} {w}" if last else w for w in words}
        return {"type": "choice", "instructions": "Next word?",
                "criteria": criteria}
    else:
        # mix: randomly v1 or v2 each call
        if random.random() < 0.5:
            return choice_q(words, reply_so_far, force_mode="v1")
        else:
            return choice_q(words, reply_so_far, force_mode="v2")


async def next_word(session, state, vocab, rng, reply_so_far=""):
    shuffled = list(vocab)
    rng.shuffle(shuffled)
    buckets = [shuffled[i:i + MAX_CHOICES] for i in range(0, len(shuffled), MAX_CHOICES)]
    groups = [buckets[i:i + QUESTIONS_PER_CALL] for i in range(0, len(buckets), QUESTIONS_PER_CALL)]

    results = await asyncio.gather(
        *(post(session, state, {f"b{gi * QUESTIONS_PER_CALL + i}": choice_q(b, reply_so_far)
                                for i, b in enumerate(g)})
          for gi, g in enumerate(groups)),
        post(session, state, {"complete": {"type": "noul", "instructions": "Is the reply complete?"}}),
    )

    complete_noul = results[-1].get("complete", {}).get("noul", 0)

    finalists = []
    for group_answers in results[:-1]:
        for ans in group_answers.values():
            if "probabilities" not in ans:
                continue
            ranked = sorted(ans["probabilities"].items(), key=lambda kv: -kv[1])
            finalists += [w for w, p in ranked[:TOP_PER_BUCKET] if p > 0]

    if END not in finalists:
        finalists.append(END)

    runoff = await post(session, state, {"final": choice_q(finalists[:MAX_CHOICES], reply_so_far)})
    probs = runoff.get("final", {}).get("probabilities", {})

    return probs, complete_noul


async def next_word_laya(session, state, vocab, rng, reply_so_far=""):
    """The same tournament, widened into questions instead of options.

    Jev scores 255 options in one question. laya-serve runs out of token budget around 150
    bare words and rejects more than 64 questions per request, so the opening sweep is
    LAYA_QUESTIONS_PER_STEP questions of LAYA_OPTIONS_PER_QUESTION words each -- 20x10 by
    default, i.e. 200 candidate words per step at ~0.28s measured -- with a runoff over the
    bucket winners exactly as the hosted path does. The cost is candidate coverage: this
    samples the vocab per step instead of scoring all of it.
    """
    nq = max(1, min(LAYA_QUESTIONS_PER_STEP, LAYA_MAX_QUESTIONS))
    k = _laya_option_budget()
    shuffled = list(vocab)
    rng.shuffle(shuffled)
    buckets = [shuffled[i:i + k] for i in range(0, min(len(shuffled), nq * k), k)]

    while True:
        questions = {f"b{i}": choice_q(b, reply_so_far) for i, b in enumerate(buckets)}
        questions["complete"] = {"type": "noul", "instructions": "Is the reply complete?"}
        status, payload, detail = await laya_post(session, state, questions)
        if status == 200:
            break
        if status in (413, 422) and len(buckets) > 1 and len(buckets[0]) > 2:
            # the option ceiling moves with the words picked, so shrink and try again,
            # and remember the size that fit for the rest of this process
            global _laya_options_fit
            buckets = [b[:max(2, len(b) // 2)] for b in buckets]
            _laya_options_fit = len(buckets[0])
            log.warning(f"laya {status} ({detail}): retrying at "
                        f"{len(buckets)}x{len(buckets[0])} options")
            continue
        log.error(f"laya request failed: {status} {detail}")
        return {}, 0.0

    complete_noul = payload.get("complete", {}).get("noul", 0)

    finalists = []
    for qid, ans in payload.items():
        if "probabilities" not in ans:
            continue
        ranked = sorted(ans["probabilities"].items(), key=lambda kv: -kv[1])
        finalists += [w for w, p in ranked[:TOP_PER_BUCKET] if p > 0]

    if END not in finalists:
        finalists.append(END)
    finalists = list(dict.fromkeys(finalists))

    status, runoff, detail = await laya_post(
        session, state, {"final": choice_q(finalists[:LAYA_MAX_OPTIONS], reply_so_far)})
    if status != 200:
        log.error(f"laya runoff failed: {status} {detail}")
    probs = runoff.get("final", {}).get("probabilities", {}) if status == 200 else {}

    return probs, complete_noul


async def generate_reply(message, history=None):
    rng = random.Random()
    vocab = vocabulary(message)
    words = []

    key = LAYA_KEY if BACKEND == "laya" else OPENROUTER_KEY
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    async with aiohttp.ClientSession(headers=headers) as session:
        for step in range(MAX_WORDS):
            turns = []
            if history:
                for h in history:
                    name = "Jev" if h["role"] == "assistant" else "User"
                    turns.append(f"{name}: {h['content']}")
            turns.append(f"User: {message}")
            turns.append(f"Jev: {render(words)}")
            state = "\n".join(turns)

            sweep = next_word_laya if BACKEND == "laya" else next_word
            probs, complete = await sweep(session, state, vocab, rng, reply_so_far=render(words))
            if not probs:
                break

            stoppable = sum(1 for w in words if w.isalnum()) >= MIN_WORDS
            if stoppable and complete >= STOP_THRESHOLD:
                log.info(f"  noul={complete:.2f} stop")
                break

            scored = {}
            for w, p in probs.items():
                if p <= 0: continue
                if w in NO_SPACE_BEFORE and words[-1:] == [w]: continue
                if w == END and not stoppable: continue
                scored[w] = p / penalty(words, w)

            if not scored: break

            ranked = sorted(scored.items(), key=lambda kv: -kv[1])
            word = ranked[0][0]

            top3 = [(w, probs.get(w, 0)) for w, _ in ranked[:3]]
            log.info(f"  [{step+1:2d}] {word:12s}  {' '.join(f'{w}:{p:.0%}' for w,p in top3)}  done={complete:.2f}")

            if word == END: break
            words.append(word)

    return render(words) if words else "..."


# ════════════════════════════════════════════════════════════════
# Discord
# ════════════════════════════════════════════════════════════════

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)
gen_lock = asyncio.Lock()

def strip_mention(c, bid):
    return re.sub(rf"<@!?{bid}>", "", c).strip()

def should_respond(m):
    if m.author.bot: return False
    if bot.user in m.mentions: return True
    if m.reference and m.reference.resolved:
        r = m.reference.resolved
        if isinstance(r, discord.Message) and r.author.id == bot.user.id: return True
    return False

@bot.event
async def on_ready():
    log.info(f"jev online as {bot.user} | vocab {len(BASE_VOCAB)} | "
             f"{'v1' if V1_MODE else 'mix'} | backend {BACKEND}")

@bot.event
async def on_message(m):
    if not should_respond(m): return
    c = strip_mention(m.content, bot.user.id) or "hello"
    log.info(f"[IN] {m.author}: {c[:80]}")
    try:
        async with m.channel.typing():
            async with gen_lock:
                h = await fetch_history(m.channel, m)
                r = await generate_reply(c, history=h)
        log.info(f"[OUT] {r}")
        await m.reply(r, mention_author=False)
    except Exception as e:
        log.error(f"Error: {e}", exc_info=True)
        try: await m.reply("...", mention_author=False)
        except: pass

if __name__ == "__main__":
    bot.run(TOKEN, log_handler=None)
