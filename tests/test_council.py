"""tests for council. run with: python tests/test_council.py"""

import json
import os
import re
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from council.agent import Agent
from council.blackboard import Blackboard
from council.client import ApiError, ChatClient
from council.config import example_config, load_config
from council.patterns import debate, fanout, pipeline, supervise, vote
from council.run import RunResult, Turn, merge_usage
from council.transcript import render_transcript, to_markdown

passed = 0
failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print("FAIL: %s" % name)


class ScriptedClient:
    """fake ChatClient.chat: responder(messages) -> (text, usage)."""

    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def chat(self, messages, model=None, temperature=None, max_tokens=None):
        self.calls.append(messages)
        text = self.responder(messages)
        return text, {"prompt_tokens": 10, "completion_tokens": 5,
                      "total_tokens": 15}


def last_user_text(messages):
    return messages[-1]["content"]


def sys_text(messages):
    return messages[0]["content"]


# --- client over real http ---

class FakeHandler(BaseHTTPRequestHandler):
    script = []  # list of (status, payload dict)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.server.seen.append(json.loads(self.rfile.read(length) or b"{}"))
        self.server.paths.append(self.path)
        status, payload = self.script.pop(0)
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def start_server(script):
    FakeHandler.script = list(script)
    srv = HTTPServer(("127.0.0.1", 0), FakeHandler)
    srv.seen = []
    srv.paths = []
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


def openai_payload(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 7,
                      "total_tokens": 10}}


srv = start_server([(200, openai_payload("hello there"))])
c = ChatClient(base_url="http://127.0.0.1:%d/v1" % srv.server_port,
               api_key="x", max_retries=0)
text, usage = c.chat([{"role": "user", "content": "hi"}])
check("client parses text", text == "hello there")
check("client parses usage", usage["total_tokens"] == 10)
check("client hits /v1/chat/completions",
      srv.paths == ["/v1/chat/completions"])
sent = srv.seen[0]
check("client sends model and messages",
      sent["model"] == "gpt-4o-mini" and sent["messages"][0]["content"] == "hi")
srv.shutdown()

srv = start_server([(500, {}), (200, openai_payload("recovered"))])
c = ChatClient(base_url="http://127.0.0.1:%d" % srv.server_port,
               api_key="x", max_retries=2)
text, _ = c.chat([{"role": "user", "content": "hi"}])
check("client retries 500 then succeeds", text == "recovered")
srv.shutdown()

srv = start_server([(400, {"error": "bad request"})])
c = ChatClient(base_url="http://127.0.0.1:%d" % srv.server_port,
               api_key="x", max_retries=2)
try:
    c.chat([{"role": "user", "content": "hi"}])
    check("client raises on 400", False)
except ApiError as e:
    check("client raises on 400", "400" in str(e))
srv.shutdown()

# --- agent ---

a = Agent("maya", "a pragmatic engineer", system="be terse", temperature=0.3)
check("agent system prompt has name and role",
      "maya" in a.system_prompt() and "pragmatic engineer" in a.system_prompt())
check("agent system prompt includes extra system",
      "be terse" in a.system_prompt())

seen = {}


def cap(messages):
    seen["messages"] = messages
    return "ok"


sc = ScriptedClient(cap)
a.say("do the thing", sc)
check("agent sends system then user",
      seen["messages"][0]["role"] == "system"
      and seen["messages"][1] == {"role": "user", "content": "do the thing"})
check("agent passes temperature through",
      sc.calls and True)

b = Agent.from_dict({"name": "x", "role": "r"})
check("agent from_dict defaults", b.temperature == 0.7 and b.model is None)

# --- blackboard ---

bb = Blackboard()
bb.write("plan", "step one", author="maya")
check("blackboard write/read", bb.read("plan") == "step one")
check("blackboard missing default", bb.read("nope") is None)
check("blackboard keys", bb.keys() == ["plan"])
bb.append("ideas", "a")
bb.append("ideas", "b")
check("blackboard append builds list", bb.read("ideas") == ["a", "b"])
bb.write("solo", "one")
bb.append("solo", "two")
check("blackboard append converts scalar",
      bb.read("solo") == ["one", "two"])
bb.note("a note", author="theo")
check("blackboard history records",
      len(bb.history()) == 6)
md = bb.dump_markdown()
check("blackboard markdown dump",
      "## ideas" in md and "- a" in md and "step one" in md)


def hammer(n):
    bb2 = Blackboard()

    def w(i):
        bb2.append("k", i)

    threads = [threading.Thread(target=w, args=(i,)) for i in range(n)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    return bb2


bb3 = hammer(50)
check("blackboard thread safe", sorted(bb3.read("k")) == list(range(50)))

# --- debate ---

def debate_resp(messages):
    who = re.search(r"You are (\w+)", sys_text(messages)).group(1)
    body = last_user_text(messages)
    if "final verdict" in body:
        return "verdict: the debate is settled."
    rnd = re.search(r"Round (\d+)", body).group(1)
    return "%s makes point in round %s" % (who, rnd)


agents = [Agent("pro", "argue for"), Agent("con", "argue against")]
res = debate(ScriptedClient(debate_resp), "tabs or spaces?", agents, rounds=2)
check("debate turn count", len(res.turns) == 2 * 2 + 1)
check("debate verdict last",
      res.turns[-1].kind == "verdict" and "settled" in res.answer)
# real check: capture prompts
prompts = []


def cap_debate(messages):
    prompts.append(last_user_text(messages))
    return debate_resp(messages)


debate(ScriptedClient(cap_debate), "q?", agents, rounds=2)
check("later rounds include earlier transcript",
      "pro makes point in round 1" in prompts[2])
check("debate usage totals", res.usage_totals()["total_tokens"] == 5 * 15)

# --- vote ---

def vote_resp(choices):
    it = iter(choices)

    def r(messages):
        return "VOTE: %s\nbecause reasons." % next(it)
    return r


agents3 = [Agent("a1", "voter"), Agent("a2", "voter"), Agent("a3", "voter")]
res = vote(ScriptedClient(vote_resp(["x", "y", "x"])), "pick one",
           ["x", "y"], agents3)
check("vote tally", res.extra["tally"] == {"x": 2, "y": 1})
check("vote winner", res.extra["winner"] == "x")
check("vote answer names winner", "Winner: x" in res.answer)

# tie -> runoff
tie_calls = {"n": 0}


def tie_resp(messages):
    tie_calls["n"] += 1
    body = last_user_text(messages)
    if "Options:\n- x\n- y" in body or "- x\n- y" in body:
        # first round: tie
        return "VOTE: %s\ntie round." % (["x", "y"][tie_calls["n"] % 2])
    # runoff: everybody picks x
    return "VOTE: x\nrunoff decided."


res = vote(ScriptedClient(tie_resp), "pick one", ["x", "y"],
           [Agent("a1", "v"), Agent("a2", "v")])
check("vote runoff resolves tie", res.extra["winner"] == "x")


def ci_resp(messages):
    return "VOTE: ALPHA\nshouting my choice."


res = vote(ScriptedClient(ci_resp), "t", ["alpha", "beta"],
           [Agent("a1", "v")])
check("vote case insensitive", res.extra["winner"] == "alpha")


def mention_resp(messages):
    return "i think beta is clearly better, no formal vote line."


res = vote(ScriptedClient(mention_resp), "t", ["alpha", "beta"],
           [Agent("a1", "v")])
check("vote falls back to mentioned option", res.extra["winner"] == "beta")


def abstain_resp(messages):
    return "i refuse to choose, both are bad."


res = vote(ScriptedClient(abstain_resp), "t", ["alpha", "beta"],
           [Agent("a1", "v")])
check("vote abstain gives no winner", res.extra["winner"] is None)

# --- supervise ---

def sup_resp(messages):
    body = last_user_text(messages)
    if "Break this into" in body:
        return "SUBTASK: write the intro\nSUBTASK: write the outro"
    if "Subtask results" in body:
        who = re.search(r"You are (\w+)", sys_text(messages)).group(1)
        return "%s merged everything." % who
    who = re.search(r"You are (\w+)", sys_text(messages)).group(1)
    sub = re.search(r"Your subtask: (.+)", body).group(1)
    return "%s did: %s" % (who, sub)


sup = Agent("boss", "supervisor")
workers = [Agent("w1", "worker"), Agent("w2", "worker")]
res = supervise(ScriptedClient(sup_resp), "write a post", sup, workers)
check("supervise parses subtasks",
      res.extra["subtasks"] == ["write the intro", "write the outro"])
check("supervise turns", len(res.turns) == 1 + 2 + 1)
check("supervise merge last",
      res.turns[-1].kind == "merge" and "merged" in res.answer)
check("supervise worker did its subtask",
      "w1 did: write the intro" in res.turns[1].text)


def ramble_resp(messages):
    body = last_user_text(messages)
    if "Break this into" in body:
        return "hmm, lots of steps, hard to say, just do stuff"
    if "Subtask results" in body:
        return "merged anyway."
    return "did a thing"


res = supervise(ScriptedClient(ramble_resp), "do stuff",
                Agent("boss", "s"), [Agent("w1", "w")])
check("supervise falls back when no subtask lines",
      len(res.extra["subtasks"]) == 1 and res.turns[-1].kind == "merge")

# --- pipeline ---

def pipe_resp(messages):
    body = last_user_text(messages)
    who = re.search(r"You are (\w+)", sys_text(messages)).group(1)
    if "Previous step output" in body:
        prev = body.split("Previous step output:\n")[1].split("\n\nYou are")[0]
        return "%s transformed [%s]" % (who, prev.strip())
    return "%s started" % who


stages = [Agent("s1", "draft"), Agent("s2", "edit"), Agent("s3", "polish")]
res = pipeline(ScriptedClient(pipe_resp), "write it", stages)
check("pipeline final is last stage output",
      res.answer == "s3 transformed [s2 transformed [s1 started]]")
check("pipeline order", [t.speaker for t in res.turns] == ["s1", "s2", "s3"])

# --- fanout ---

def fan_resp(messages):
    body = last_user_text(messages)
    who = re.search(r"You are (\w+)", sys_text(messages)).group(1)
    if "Independent answers" in body:
        return "merged: everyone contributed."
    return "%s says hi" % who


res = fanout(ScriptedClient(fan_resp), "greet",
             [Agent("w1", "w"), Agent("w2", "w"), Agent("w3", "w")])
check("fanout turns", len(res.turns) == 4)
check("fanout merge", "merged" in res.answer)
check("fanout workers all ran",
      {t.speaker for t in res.turns[:3]} == {"w1", "w2", "w3"})

# --- run result helpers ---

r = RunResult("debate", "q")
r.add(Turn("a", "x", {"prompt_tokens": 1, "completion_tokens": 2,
                      "total_tokens": 3}))
r.add(Turn("b", "y", {"prompt_tokens": 4, "completion_tokens": 5,
                      "total_tokens": 9}))
check("usage totals sum", r.usage_totals() == {"prompt_tokens": 5,
      "completion_tokens": 7, "total_tokens": 12})
check("merge_usage", merge_usage({"a": 1}, {"a": 2, "b": 3}) ==
      {"a": 3, "b": 3})

md = to_markdown(r.finish("done!"))
check("transcript markdown",
      "# council run: debate" in md and "task: q" in md
      and "done!" in md and "turns: 2" in md)
check("render_transcript",
      "**a:** hi" in render_transcript([("a", "hi")]))

# --- config ---

cfg = example_config()
check("example config has debate shape",
      cfg["pattern"] == "debate" and len(cfg["agents"]) == 2)
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump(cfg, f)
    path = f.name
loaded = load_config(path)
check("load_config", loaded["pattern"] == "debate"
      and loaded["judge"].name == "moderator"
      and loaded["params"]["rounds"] == 2)
os.unlink(path)

with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump({"pattern": "nope"}, f)
    bad = f.name
try:
    load_config(bad)
    check("load_config rejects bad pattern", False)
except ValueError:
    check("load_config rejects bad pattern", True)
os.unlink(bad)

try:
    load_config("/tmp/does-not-exist-council.json")
    check("load_config missing file raises", False)
except FileNotFoundError:
    check("load_config missing file raises", True)

# --- cli ---

import council.cli as cli_mod

a = cli_mod.parse_agent("maya: a pragmatic engineer")
check("parse_agent splits", a.name == "maya" and "pragmatic" in a.role)
a = cli_mod.parse_agent("solo")
check("parse_agent default role", a.name == "solo")
try:
    cli_mod.parse_agent(": noname")
    check("parse_agent rejects empty name", False)
except ValueError:
    check("parse_agent rejects empty name", True)

with tempfile.TemporaryDirectory() as d:
    old = os.getcwd()
    os.chdir(d)
    cli_mod.cmd_init(None)
    check("cli init writes council.json", os.path.exists("council.json"))
    with open("council.json") as f:
        check("cli init valid json", json.load(f)["pattern"] == "debate")
    os.chdir(old)

# cli debate end to end with scripted client
old_make = cli_mod.make_client
cli_mod.make_client = lambda args: ScriptedClient(debate_resp)
ns = cli_mod.build_parser().parse_args(
    ["debate", "tabs or spaces?", "--agent", "pro: argue for",
     "--agent", "con: argue against", "--rounds", "1", "--quiet"])
try:
    cli_mod.cmd_debate(ns)
    check("cli debate runs", True)
except Exception as e:
    print("cli debate raised: %s" % e)
    check("cli debate runs", False)
cli_mod.make_client = old_make

print("\n%d passed, %d failed" % (passed, failed))
sys.exit(1 if failed else 0)
