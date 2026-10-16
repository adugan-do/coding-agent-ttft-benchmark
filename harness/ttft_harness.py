#!/usr/bin/env python3
"""TTFT harness for coding-agent sessions on DigitalOcean Serverless Inference.

Subcommands:
  models                       list models on the endpoint
  probe     --model M          one tiny streaming call: checks stream usage + cached_tokens fields
  run       [options]          run experiments, append raw rows to a CSV
  summarize --csv FILE         p50/p95/p99, bootstrap CI of the p50, error counts, hit rate
  cost      --csv FILE --prices P  dollar cost per request and per session, with vs without caching

Environment:
  MODEL_ACCESS_KEY   DigitalOcean model access key (required)
  DO_BASE_URL        override base URL (default https://inference.do-ai.run/v1)
"""
import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import os
import random
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

TOOL_CHOICE = "auto"  # "none" makes some models drop tool definitions from the prompt
TEMPERATURE = 0.0     # fixed for every request; None omits the parameter
MAX_TOKENS = 256      # output cap; reasoning models can spend all of it before any content token
BASE_URL = os.environ.get("DO_BASE_URL", "https://inference.do-ai.run/v1")

FIELDS = [
    "run_id", "experiment", "variant", "phase", "model", "target_tokens",
    "prompt_tokens", "cached_tokens", "completion_tokens", "hit_rate", "ttft_first_chunk_s",
    "ttft_content_s", "total_s", "turn", "concurrency", "region",
    "timestamp_utc", "served_model", "error",
]

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from the repository.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run the test suite or a single test.",
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "grep", "description": "Search the repository for a regex.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace text in a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old": {"type": "string"},
            "new": {"type": "string"}}, "required": ["path", "old", "new"]}}},
]

SYSTEM_PROMPT = (
    "You are a coding agent working inside a Python repository. Use the provided "
    "tools to inspect files, run tests, and make small, correct edits. Think about "
    "the smallest change that fixes the problem, verify it with the tests, and keep "
    "answers brief. Never invent file contents; read them first."
)

WORDS = ("def class return import self value result config handler request response "
         "client session token cache prefix index parse build load store update "
         "render format error retry timeout buffer stream queue worker task "
         "schema model field query batch commit merge split join").split()


# --------------------------------------------------------------------------
# Synthetic content
# --------------------------------------------------------------------------
def fake_code(rng, n_chars):
    """Deterministic code-looking text of about n_chars characters."""
    lines, size = [], 0
    while size < n_chars:
        indent = "    " * rng.randint(0, 3)
        a, b, c = (rng.choice(WORDS) for _ in range(3))
        kind = rng.random()
        if kind < 0.25:
            line = f"{indent}def {a}_{b}(self, {c}, retries={rng.randint(0, 9)}):"
        elif kind < 0.5:
            line = f"{indent}{a}_{b} = self.{c}.get('{a}', {rng.randint(0, 9999)})"
        elif kind < 0.7:
            line = f"{indent}if {a}_{b} is None: raise ValueError('{c} {rng.randint(0, 999)}')"
        elif kind < 0.85:
            line = f"{indent}# {a} {b} {c} handled at line {rng.randint(1, 4000)}"
        else:
            line = f"{indent}return {a}({b}, {c}={rng.randint(0, 99)})"
        lines.append(line)
        size += len(line) + 1
    return "\n".join(lines)


def tool_pair(seed, step, n_chars):
    """One agent step: assistant tool call + tool result."""
    rng = random.Random(f"{seed}:{step}")
    call_id = "call_" + hashlib.md5(f"{seed}:{step}".encode()).hexdigest()[:12]
    path = f"src/{rng.choice(WORDS)}/{rng.choice(WORDS)}_{step}.py"
    assistant = {"role": "assistant", "content": None, "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": "read_file",
                     "arguments": json.dumps({"path": path}, sort_keys=True)}}]}
    tool = {"role": "tool", "tool_call_id": call_id,
            "content": f"# {path}\n" + fake_code(rng, n_chars)}
    return [assistant, tool]


def build_messages(system_text, seed, n_steps, chars_per_step, extra_steps=0, extra_chars=600):
    """system + user task + base history (stable) + optional extra appended steps.

    Prefix stability: base history depends only on (seed, step), so every call that
    shares seed/system_text shares a byte-identical prefix, and `extra_steps` only
    appends.
    """
    msgs = [{"role": "system", "content": system_text},
            {"role": "user", "content": "The test test_retry_backoff fails. Find the cause and fix it."}]
    for step in range(n_steps):
        msgs += tool_pair(seed, step, chars_per_step)
    for k in range(extra_steps):
        msgs += tool_pair(seed, 10_000 + k, extra_chars)
    # Constant tail: forces a plain-text reply so TTFT is observable as streamed text.
    msgs.append({"role": "user", "content":
                 "In one short sentence, say what you have learned so far. Do not call any tools."})
    return msgs


# --------------------------------------------------------------------------
# Streaming call
# --------------------------------------------------------------------------
def make_client():
    key = os.environ.get("MODEL_ACCESS_KEY")
    if not key:
        sys.exit("MODEL_ACCESS_KEY is not set")
    return OpenAI(base_url=BASE_URL, api_key=key, timeout=300, max_retries=0)


def stream_call(client, model, messages, tools=None, max_tokens=None, extra_body=None, extra_headers=None):
    """Return dict with TTFT (first chunk and first content token), usage, total time."""
    kwargs = dict(model=model, messages=messages, max_tokens=max_tokens or MAX_TOKENS, stream=True,
                  stream_options={"include_usage": True})
    if TEMPERATURE is not None:
        kwargs["temperature"] = TEMPERATURE
    if tools:
        kwargs["tools"] = tools
        if TOOL_CHOICE != "auto":
            kwargs["tool_choice"] = TOOL_CHOICE
    if extra_body:
        kwargs["extra_body"] = extra_body
    if extra_headers:
        kwargs["extra_headers"] = extra_headers
    out = {"ttft_first_chunk_s": None, "ttft_content_s": None, "total_s": None,
           "prompt_tokens": None, "cached_tokens": None, "completion_tokens": None, "served_model": "", "error": ""}
    start = time.perf_counter()
    try:
        for chunk in client.chat.completions.create(**kwargs):
            now = time.perf_counter() - start
            if not out["served_model"] and getattr(chunk, "model", None):
                out["served_model"] = chunk.model
            if chunk.choices:
                delta = chunk.choices[0].delta
                has_content = bool(getattr(delta, "content", None))
                has_tool = bool(getattr(delta, "tool_calls", None))
                has_reason = bool(getattr(delta, "reasoning_content", None)
                                  or getattr(delta, "reasoning", None))
                if out["ttft_first_chunk_s"] is None and (has_content or has_reason or has_tool):
                    out["ttft_first_chunk_s"] = now
                if out["ttft_content_s"] is None and has_content:
                    out["ttft_content_s"] = now
            usage = getattr(chunk, "usage", None)
            if usage:
                out["prompt_tokens"] = usage.prompt_tokens
                out["completion_tokens"] = getattr(usage, "completion_tokens", None)
                det = getattr(usage, "prompt_tokens_details", None)
                vals = [getattr(det, "cached_tokens", None) if det is not None else None,
                        getattr(usage, "cache_read_input_tokens", None)]
                vals = [v for v in vals if v is not None]
                # Some models report cache reads in one field, some in the other.
                out["cached_tokens"] = max(vals) if vals else None
        out["total_s"] = time.perf_counter() - start
    except Exception as e:  # record, don't crash a long run
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["total_s"] = time.perf_counter() - start
    return out


# --------------------------------------------------------------------------
# Raw-row writer
# --------------------------------------------------------------------------
class Recorder:
    def __init__(self, path, region, max_requests):
        self.path, self.region = path, region
        self.max_requests, self.count = max_requests, 0
        self.lock = threading.Lock()
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=FIELDS)
        if new:
            self.w.writeheader()

    def budget_left(self):
        return self.count < self.max_requests

    def record(self, res, **meta):
        pt, ct = res["prompt_tokens"], res["cached_tokens"]
        row = {k: "" for k in FIELDS}
        row.update(meta)
        row.update({
            "run_id": uuid.uuid4().hex[:10],
            "prompt_tokens": pt if pt is not None else "",
            "cached_tokens": ct if ct is not None else "",
            "completion_tokens": res["completion_tokens"] if res["completion_tokens"] is not None else "",
            "hit_rate": round(ct / pt, 4) if pt and ct is not None else "",
            "ttft_first_chunk_s": _r(res["ttft_first_chunk_s"]),
            "ttft_content_s": _r(res["ttft_content_s"]),
            "total_s": _r(res["total_s"]),
            "region": self.region,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "served_model": res.get("served_model", ""),
            "error": res["error"],
        })
        with self.lock:
            self.count += 1
            self.w.writerow(row)
            self.f.flush()
        return row


def _r(x):
    return round(x, 4) if x is not None else ""


# --------------------------------------------------------------------------
# Calibration: tokens per character for each model's tokenizer
# --------------------------------------------------------------------------
def calibrate(client, model):
    """Estimate prompt tokens per character of our synthetic code. Uses two unique
    (uncached) probes so the fixed template overhead cancels out."""
    pts = []
    for n_chars in (4_000, 40_000):
        salt = uuid.uuid4().hex
        msgs = build_messages(f"{SYSTEM_PROMPT}\n[probe {salt}]", salt, 1, n_chars)
        r = stream_call(client, model, msgs, tools=TOOLS, max_tokens=1)
        if r["prompt_tokens"] is None:
            raise RuntimeError(f"{model}: no usage in stream ({r['error'] or 'usage missing'})")
        pts.append((n_chars, r["prompt_tokens"]))
    (c1, t1), (c2, t2) = pts
    tpc = (t2 - t1) / (c2 - c1)
    fixed = t1 - tpc * c1
    return tpc, fixed


def size_session(target_tokens, tpc, fixed, chars_per_step=6000):
    """Return n_steps so the base session is about target_tokens."""
    per_step = tpc * chars_per_step + 30  # +30 for call/message framing
    return max(1, int(round((target_tokens - fixed) / per_step))), chars_per_step


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def exp_cold_warm(client, rec, model, target, tpc, fixed, cold_n, warm_n, sleep, interleave=True):
    """Cold calls (unique prefix every time) and warm calls (one growing session).

    With interleave=True the cold and warm calls are shuffled together, so slow periods on the
    platform hit both conditions. Otherwise all cold calls run first, then the warm session.
    """
    n_steps, cps = size_session(target, tpc, fixed)
    salt = uuid.uuid4().hex
    system = f"[salt {salt}] {SYSTEM_PROMPT}"

    def cold_call(i):
        # Cold: unique system prompt salt invalidates the whole prefix.
        csalt = uuid.uuid4().hex
        msgs = build_messages(f"[salt {csalt}] {SYSTEM_PROMPT}", csalt, n_steps, cps)
        res = stream_call(client, model, msgs, tools=TOOLS)
        rec.record(res, experiment="cold_warm", variant="stable", phase="cold",
                   model=model, target_tokens=target, turn=i, concurrency=1)

    def warm_call(t):
        # Warm: turn 0 primes the cache, then each call appends one step to the same prefix.
        msgs = build_messages(system, salt, n_steps, cps, extra_steps=t)
        res = stream_call(client, model, msgs, tools=TOOLS)
        rec.record(res, experiment="cold_warm", variant="stable",
                   phase="prime" if t == 0 else "warm", model=model,
                   target_tokens=target, turn=t, concurrency=1)

    if interleave:
        mixed = ["cold"] * cold_n + ["warm"] * warm_n
        random.Random(salt).shuffle(mixed)
        order = ["prime"] + mixed
    else:
        order = ["cold"] * cold_n + ["prime"] + ["warm"] * warm_n
    i = t = 0
    for kind in order:
        if not rec.budget_left():
            return
        if kind == "cold":
            cold_call(i)
            i += 1
        elif kind == "prime":
            warm_call(0)
        else:
            t += 1
            warm_call(t)
        time.sleep(sleep)


PERMS = list(itertools.permutations(TOOLS))


def exp_cache_break(client, rec, model, target, tpc, fixed, n, sleep, variants=None):
    """Same session shape under prefix variants.

    stable: identical prefix. timestamp: current time in the system prompt.
    reordered_alt: tools alternate between two orders (legacy; both orders get cached).
    reordered_unique: a different tool order on every call (24 permutations).
    stable_key: stable prefix plus a constant prompt_cache_key (DigitalOcean's documented usage).
    changing_key: stable prefix but a new prompt_cache_key on every call (the documented anti-pattern).
    """
    n_steps, cps = size_session(target, tpc, fixed)
    for variant in (variants or ("stable", "timestamp", "reordered_unique")):
        salt = uuid.uuid4().hex
        for t in range(n + 1):
            if not rec.budget_left():
                return
            system = f"[salt {salt}] {SYSTEM_PROMPT}"
            tools = TOOLS
            extra_body = None
            if variant == "stable_key":
                extra_body = {"prompt_cache_key": f"ttft-{salt[:16]}"}
            elif variant == "changing_key":
                extra_body = {"prompt_cache_key": f"ttft-{uuid.uuid4().hex[:16]}"}
            if variant == "timestamp":
                system += f"\nCurrent time: {dt.datetime.now(dt.timezone.utc).isoformat()}"
            elif variant == "reordered_alt":
                tools = TOOLS[::-1] if t % 2 else TOOLS
            elif variant == "reordered_unique":
                tools = list(PERMS[t % len(PERMS)])
            msgs = build_messages(system, salt, n_steps, cps, extra_steps=t)
            res = stream_call(client, model, msgs, tools=tools, extra_body=extra_body)
            rec.record(res, experiment="cache_break", variant=variant,
                       phase="prime" if t == 0 else "warm", model=model,
                       target_tokens=target, turn=t, concurrency=1)
            time.sleep(sleep)


def exp_parallel(client, rec, model, target, tpc, fixed, width, rounds, sleep):
    """Prime a prefix, then fire `width` concurrent requests that all share it."""
    n_steps, cps = size_session(target, tpc, fixed)
    salt = uuid.uuid4().hex
    system = f"[salt {salt}] {SYSTEM_PROMPT}"
    base = build_messages(system, salt, n_steps, cps)
    res = stream_call(client, model, base, tools=TOOLS)
    rec.record(res, experiment="parallel", variant="stable", phase="prime",
               model=model, target_tokens=target, turn=0, concurrency=1)
    for r in range(1, rounds + 1):
        if not rec.budget_left():
            return
        def one(k):
            msgs = build_messages(system, salt, n_steps, cps, extra_steps=1)
            msgs[-1]["content"] += f"\n# branch {r}.{k}"  # diverge only at the very end
            return stream_call(client, model, msgs, tools=TOOLS)
        with ThreadPoolExecutor(max_workers=width) as ex:
            results = list(ex.map(one, range(width)))
        for res in results:
            rec.record(res, experiment="parallel", variant="stable", phase="warm",
                       model=model, target_tokens=target, turn=r, concurrency=width)
        time.sleep(sleep)


def exp_router(client, rec, target, tpc, fixed, cold_n, warm_n, sleep):
    """Send the same session shape to inference-router under each routing preference.
    The served model is whatever the router picks (see served_model column)."""
    n_steps, cps = size_session(target, tpc, fixed)
    variants = [("balanced", {"routing_preference": "balanced"}, None),
                ("budget", {"routing_preference": "budget"}, None),
                ("quality", {"routing_preference": "quality"}, None),
                ("balanced_affinity_none", {"routing_preference": "balanced"},
                 {"X-Model-Affinity": "none"})]
    for name, body, headers in variants:
        for i in range(cold_n):
            if not rec.budget_left():
                return
            salt = uuid.uuid4().hex
            msgs = build_messages(f"[salt {salt}] {SYSTEM_PROMPT}", salt, n_steps, cps)
            res = stream_call(client, "inference-router", msgs, tools=TOOLS,
                              extra_body=body, extra_headers=headers)
            rec.record(res, experiment="router", variant=name, phase="cold",
                       model="inference-router", target_tokens=target, turn=i, concurrency=1)
            time.sleep(sleep)
        salt = uuid.uuid4().hex
        system = f"[salt {salt}] {SYSTEM_PROMPT}"
        for t in range(warm_n + 1):
            if not rec.budget_left():
                return
            msgs = build_messages(system, salt, n_steps, cps, extra_steps=t)
            res = stream_call(client, "inference-router", msgs, tools=TOOLS,
                              extra_body=body, extra_headers=headers)
            rec.record(res, experiment="router", variant=name,
                       phase="prime" if t == 0 else "warm", model="inference-router",
                       target_tokens=target, turn=t, concurrency=1)
            time.sleep(sleep)


def exp_baseline(client, rec, model, n, sleep):
    """Tiny uncached prompt: the fixed floor (network, queueing, first token) with almost no prefill."""
    for i in range(n):
        if not rec.budget_left():
            return
        msgs = [{"role": "user", "content": f"[{uuid.uuid4().hex}] Reply with the single word: ok"}]
        res = stream_call(client, model, msgs, max_tokens=8)
        rec.record(res, experiment="baseline", variant="tiny_prompt", phase="cold",
                   model=model, target_tokens=0, turn=i, concurrency=1)
        time.sleep(sleep)


def exp_rtt(client, rec, n, sleep):
    """Round trip to the API host with no inference (lists models, costs no tokens). Shows how much
    of any TTFT number is distance and auth overhead from this client."""
    for i in range(n):
        if not rec.budget_left():
            return
        start, err = time.perf_counter(), ""
        try:
            client.models.list()
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
        el = time.perf_counter() - start
        res = {"ttft_first_chunk_s": el, "ttft_content_s": None, "total_s": el, "prompt_tokens": None,
               "cached_tokens": None, "completion_tokens": None, "served_model": "", "error": err}
        rec.record(res, experiment="baseline", variant="rtt_models_list", phase="cold",
                   model="(none)", target_tokens=0, turn=i, concurrency=1)
        time.sleep(sleep)


# --------------------------------------------------------------------------
# Cost / request estimate
# --------------------------------------------------------------------------
def estimate(args):
    models, sizes = args.models, args.sizes
    reqs, tokens = 0, 0
    for _ in models:
        reqs += 2  # calibration
        tokens += 44_000 // 3
        if "baseline" in args.experiments:
            reqs += args.baseline_n
            tokens += args.baseline_n * 50
        for s in sizes:
            if "cold_warm" in args.experiments:
                n = args.cold_n + args.warm_n + 1
                reqs += n
                tokens += args.cold_n * s + (args.warm_n + 1) * s
        if "cache_break" in args.experiments:  # runs once per model, at --break-size
            n = len(args.break_variants or (1, 2, 3)) * (args.cache_break_n + 1)
            reqs += n
            tokens += n * args.break_size
        if "parallel" in args.experiments:
            n = 1 + args.parallel_rounds * args.parallel_width
            reqs += n
            tokens += n * args.parallel_size
    if "router" in args.experiments:
        n = 4 * (args.router_cold_n + args.router_warm_n + 1)
        reqs += n + 2
        tokens += n * args.router_size
    return reqs, tokens


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def cmd_models(_):
    for m in make_client().models.list().data:
        print(m.id)


def cmd_probe(args):
    c = make_client()
    msgs = build_messages(SYSTEM_PROMPT + f"\n[probe {uuid.uuid4().hex}]", "probe", 2, 3000)
    for i in range(2):  # second identical call should show cached_tokens if caching is active
        r = stream_call(c, args.model, msgs, tools=TOOLS, max_tokens=32)
        print(f"call {i + 1}: {json.dumps(r)}")
        time.sleep(1.5)


def cmd_run(args):
    global TOOL_CHOICE, TEMPERATURE, MAX_TOKENS
    TOOL_CHOICE = args.tool_choice
    TEMPERATURE = args.temperature if args.temperature >= 0 else None
    MAX_TOKENS = args.max_tokens
    reqs, toks = estimate(args)
    print(f"Plan: ~{reqs} requests, ~{toks / 1e6:.1f}M prompt tokens (upper bound, ignores cache discount)")
    if args.dry_run:
        return
    if reqs > args.max_requests:
        print(f"note: plan exceeds --max-requests {args.max_requests}; the run will stop at the cap")
    client = make_client()
    rec = Recorder(args.out, args.region, args.max_requests)
    if "baseline" in args.experiments:
        print("== network round trip (GET /models, no tokens)")
        exp_rtt(client, rec, args.baseline_n, args.sleep)
    for model in args.models:
        print(f"== {model}: calibrating")
        try:
            tpc, fixed = calibrate(client, model)
        except Exception as e:
            print(f"   skip {model}: {e}")
            continue
        print(f"   tokens/char={tpc:.4f}, fixed overhead={fixed:.0f} tokens")
        if "baseline" in args.experiments:
            print(f"   baseline (tiny prompt) x{args.baseline_n}")
            exp_baseline(client, rec, model, args.baseline_n, args.sleep)
        if "cold_warm" in args.experiments:
            for s in args.sizes:
                print(f"   cold_warm {s}")
                exp_cold_warm(client, rec, model, s, tpc, fixed, args.cold_n, args.warm_n, args.sleep,
                              interleave=not args.no_interleave)
        if "cache_break" in args.experiments:
            print(f"   cache_break {args.break_size}")
            exp_cache_break(client, rec, model, args.break_size, tpc, fixed, args.cache_break_n, args.sleep,
                            variants=args.break_variants)
        if "parallel" in args.experiments:
            print(f"   parallel x{args.parallel_width} at {args.parallel_size}")
            exp_parallel(client, rec, model, args.parallel_size, tpc, fixed,
                         args.parallel_width, args.parallel_rounds, args.sleep)
    if "router" in args.experiments:
        print(f"== inference-router: calibrating on {args.router_calib_model}")
        tpc, fixed = calibrate(client, args.router_calib_model)
        print(f"   router at {args.router_size}")
        exp_router(client, rec, args.router_size, tpc, fixed,
                   args.router_cold_n, args.router_warm_n, args.sleep)
    print(f"wrote {rec.count} rows to {args.out}")


def pct(xs, p):
    xs = sorted(xs)
    return xs[int(round(p / 100 * (len(xs) - 1)))] if xs else float("nan")


def boot_ci_median(xs, reps=1000, seed=0):
    """95% bootstrap confidence interval for the median."""
    if len(xs) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(xs, k=len(xs))) for _ in range(reps))
    return meds[int(0.025 * reps)], meds[int(0.975 * reps) - 1]


def cmd_cost(args):
    """Dollar cost from recorded usage. --prices is a JSON file of per-1M-token prices you copy from
    the pricing page: {"kimi-k3": {"input": 3.0, "cached_input": 0.3, "output": 0.0}, ...}
    Completion tokens (which include reasoning tokens, if the model reports them that way) are priced
    at "output". Models missing from the file are skipped."""
    with open(args.prices) as f:
        prices = json.load(f)
    groups, missing = {}, set()
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            if r["error"] or not r["prompt_tokens"]:
                continue
            p = prices.get(r["model"])
            if p is None:
                missing.add(r["model"])
                continue
            pt = int(r["prompt_tokens"])
            ct = int(float(r["cached_tokens"] or 0))
            out = int(float(r["completion_tokens"] or 0))
            actual = ((pt - ct) * p["input"] + ct * p["cached_input"] + out * p["output"]) / 1e6
            nocache = (pt * p["input"] + out * p["output"]) / 1e6
            key = (r["experiment"], r["variant"], r["phase"], r["model"], r["target_tokens"])
            groups.setdefault(key, []).append((actual, nocache))
    if missing:
        print(f"note: no prices for {sorted(missing)}; those rows are skipped", file=sys.stderr)
    print("experiment,variant,phase,model,target,n,mean_cost_usd,mean_cost_no_cache_usd,savings_pct")
    means = {}
    for key in sorted(groups):
        a = statistics.mean(x[0] for x in groups[key])
        b = statistics.mean(x[1] for x in groups[key])
        means[key] = (a, b)
        print(",".join(map(str, key)) + f",{len(groups[key])},{a:.5f},{b:.5f},{(1 - a / b) * 100 if b else 0:.1f}")
    n = args.session_turns
    print(f"\nsession estimate: {n} calls = 1 cold call + {n - 1} warm calls (stable prefix)")
    print("model,target,session_cost_usd,session_cost_no_cache_usd,savings_pct")
    for (exp, var, phase, model, target), (cold_a, cold_b) in sorted(means.items()):
        if (exp, var, phase) != ("cold_warm", "stable", "cold"):
            continue
        warm = means.get(("cold_warm", "stable", "warm", model, target))
        if warm is None:
            continue
        s_a = cold_a + (n - 1) * warm[0]
        s_b = cold_b + (n - 1) * warm[1]
        print(f"{model},{target},{s_a:.4f},{s_b:.4f},{(1 - s_a / s_b) * 100 if s_b else 0:.1f}")


def cmd_summarize(args):
    groups, errs = {}, {}
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        # A merged CSV can carry a "source" column; keep each source's cells separate.
        has_source = "source" in (reader.fieldnames or [])
        for r in reader:
            key = (r["experiment"], r["variant"], r["phase"], r["model"], r["target_tokens"])
            if has_source:
                key = (r["source"],) + key
            if r["error"]:
                errs[key] = errs.get(key, 0) + 1
                continue
            groups.setdefault(key, []).append(r)
    if args.served:
        c = {}
        with open(args.csv) as f:
            for r in csv.DictReader(f):
                if r["experiment"] == "router" and r["served_model"]:
                    k = (r["variant"], r["served_model"])
                    c[k] = c.get(k, 0) + 1
        for k in sorted(c):
            print("served", k[0], k[1], c[k])
    # p95/p99 are only reported as reliable with at least 100 samples (tail_ok).
    print(("source," if has_source else "") + "experiment,variant,phase,model,target,n,p50_ms,p95_ms,p99_ms,"
          "content_p50_ms,median_hit_rate,median_prompt_tokens,"
          "p50_ci95_lo_ms,p50_ci95_hi_ms,n_errors,tail_ok")
    for key in sorted(set(groups) | set(errs)):
        rows = groups.get(key, [])
        ttft = [float(r["ttft_first_chunk_s"]) * 1000 for r in rows if r["ttft_first_chunk_s"]]
        ttc = [float(r["ttft_content_s"]) * 1000 for r in rows if r["ttft_content_s"]]
        hr = [float(r["hit_rate"]) for r in rows if r["hit_rate"] != ""]
        pt = [int(r["prompt_tokens"]) for r in rows if r["prompt_tokens"]]
        lo, hi = boot_ci_median(ttft)
        print(",".join(map(str, key)) + f",{len(rows)},{pct(ttft, 50):.0f},{pct(ttft, 95):.0f},"
              f"{pct(ttft, 99):.0f},{pct(ttc, 50):.0f},"
              f"{statistics.median(hr) if hr else float('nan'):.3f},"
              f"{statistics.median(pt) if pt else 0:.0f},"
              f"{lo:.0f},{hi:.0f},{errs.get(key, 0)},{'yes' if len(ttft) >= 100 else 'no'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models").set_defaults(fn=cmd_models)
    pp = sub.add_parser("probe")
    pp.add_argument("--model", required=True)
    pp.set_defaults(fn=cmd_probe)
    r = sub.add_parser("run")
    r.add_argument("--models", nargs="*", default=[])
    r.add_argument("--sizes", nargs="+", type=int, default=[20_000, 60_000, 120_000])
    r.add_argument("--experiments", nargs="+", default=["baseline", "cold_warm", "cache_break", "parallel"],
                   choices=["baseline", "cold_warm", "cache_break", "parallel", "router"])
    r.add_argument("--baseline-n", type=int, default=10,
                   help="tiny-prompt calls per model, and network round trips (no tokens)")
    r.add_argument("--temperature", type=float, default=0.0,
                   help="sent with every request; use a negative value to omit the parameter")
    r.add_argument("--no-interleave", action="store_true",
                   help="run all cold calls before the warm session instead of shuffling them together")
    r.add_argument("--max-tokens", type=int, default=256,
                   help="output cap per request; raise it for reasoning models so content tokens are reached")
    r.add_argument("--cold-n", type=int, default=10)
    r.add_argument("--warm-n", type=int, default=30)
    r.add_argument("--cache-break-n", type=int, default=10)
    r.add_argument("--break-size", type=int, default=60_000)
    r.add_argument("--break-variants", nargs="+", default=None,
                   choices=["stable", "timestamp", "reordered_alt", "reordered_unique",
                            "stable_key", "changing_key"])
    r.add_argument("--parallel-width", type=int, default=8)
    r.add_argument("--parallel-rounds", type=int, default=5)
    r.add_argument("--parallel-size", type=int, default=60_000)
    r.add_argument("--router-size", type=int, default=60_000)
    r.add_argument("--router-cold-n", type=int, default=3)
    r.add_argument("--router-warm-n", type=int, default=15)
    r.add_argument("--router-calib-model", default="qwen3.8-max",
                   help="direct model used only to calibrate tokens/char for router sessions")
    r.add_argument("--sleep", type=float, default=0.5)
    r.add_argument("--region", default="unknown")
    r.add_argument("--tool-choice", default="auto", choices=["auto", "none"])
    r.add_argument("--out", default="results.csv")
    r.add_argument("--max-requests", type=int, default=2000)
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("summarize")
    s.add_argument("--csv", required=True)
    s.add_argument("--served", action="store_true", help="also print router served-model counts")
    s.set_defaults(fn=cmd_summarize)
    c = sub.add_parser("cost")
    c.add_argument("--csv", required=True)
    c.add_argument("--prices", required=True,
                   help='JSON: {"model": {"input": usd_per_1M, "cached_input": usd_per_1M, "output": usd_per_1M}}')
    c.add_argument("--session-turns", type=int, default=50)
    c.set_defaults(fn=cmd_cost)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
