"""tests for council tool calling. run with: python tests/test_tools.py"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from council.agent import Agent, BudgetExceeded
from council.blackboard import Blackboard
from council.client import ChatClient
from council.patterns import debate, supervise
from council.tools import (CALC, Tool, ToolError, blackboard_tools, get,
                           register, resolve)

passed = 0
failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print("FAIL: %s" % name)


def calc_call(expr, cid="call_1"):
    return [{"id": cid, "name": "calc", "arguments": {"expr": expr},
             "arguments_raw": json.dumps({"expr": expr})}]


class ToolScriptedClient:
    """fake client: script entries are (text, calls); calls None for chat."""

    def __init__(self, script):
        self.script = list(script)
        self.chat_calls = 0
        self.tools_calls = 0
        self.sent_tools = []

    def _usage(self):
        return {"prompt_tokens": 10, "completion_tokens": 5,
                "total_tokens": 15}

    def chat(self, messages, model=None, temperature=None, max_tokens=None):
        self.chat_calls += 1
        return self.script.pop(0)[0], self._usage()

    def chat_tools(self, messages, tools, model=None, temperature=None,
                   max_tokens=None):
        self.tools_calls += 1
        self.sent_tools.append(
            [t["function"]["name"] for t in tools] if tools else None)
        text, calls = self.script.pop(0)
        return text, self._usage(), calls or []


# --- tool basics ---

s = CALC.schema()
check("calc schema shape",
      s["type"] == "function" and s["function"]["name"] == "calc"
      and s["function"]["parameters"]["required"] == ["expr"])
check("calc math", CALC.run({"expr": "(12 + 34) * 2"}) == "92")
check("calc rejects code",
      CALC.run({"expr": "__import__('os').system('x')"}).startswith("error:"))
check("calc div by zero is an error",
      CALC.run({"expr": "1/0"}).startswith("error:"))
check("run with non-dict args errors",
      CALC.run("2+2").startswith("error:"))
check("run with missing arg reports bad arguments",
      CALC.run({}).startswith("error: bad arguments"))
check("none result becomes ok",
      Tool("n", "d", {}, lambda: None).run({}) == "ok")


def boom():
    raise ValueError("kaput")


check("tool exception becomes error text",
      Tool("b", "d", {}, boom).run({}) == "error: ValueError: kaput")


def clean_fail():
    raise ToolError("nope")


check("ToolError message passes through clean",
      Tool("c", "d", {}, clean_fail).run({}) == "error: nope")

check("resolve calc", resolve(["calc"])[0] is CALC)
check("resolve accepts a bare string", resolve("calc")[0] is CALC)

custom = Tool("shout", "uppercases", {"type": "object",
              "properties": {"s": {"type": "string"}},
              "required": ["s"]}, lambda s: s.upper())
register(custom)
check("register/get roundtrip", get("shout") is custom)
check("custom tool runs", get("shout").run({"s": "hi"}) == "HI")
try:
    get("does-not-exist")
    check("get unknown raises", False)
except ValueError as e:
    check("get unknown raises", "does-not-exist" in str(e))
try:
    resolve(["calc", "does-not-exist"])
    check("resolve unknown raises", False)
except ValueError:
    check("resolve unknown raises", True)

# --- blackboard tools ---

bb = Blackboard()
bread, bwrite = blackboard_tools(bb, author="maya")
check("blackboard write tool", bwrite.run({"key": "plan", "value": "v1"}) == "wrote 'plan'")
check("blackboard read tool", bread.run({"key": "plan"}) == "v1")
check("blackboard read missing",
      bread.run({"key": "nope"}).startswith("no value stored"))
check("blackboard tool wrote with author",
      bb.history()[0][0] == "maya")

# --- client chat_tools over real http ---

class FakeHandler(BaseHTTPRequestHandler):
    script = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.server.seen.append(json.loads(self.rfile.read(length) or b"{}"))
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
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


tool_payload = {
    "choices": [{"message": {"role": "assistant", "content": "",
                 "tool_calls": [{"id": "call_9", "type": "function",
                                 "function": {"name": "calc",
                                              "arguments": '{"expr": "2+2"}'}}]}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10},
}
srv = start_server([(200, tool_payload)])
c = ChatClient(base_url="http://127.0.0.1:%d/v1" % srv.server_port,
               api_key="x", max_retries=0)
text, usage, calls = c.chat_tools([{"role": "user", "content": "hi"}],
                                  [CALC.schema()])
check("chat_tools parses call",
      len(calls) == 1 and calls[0]["name"] == "calc"
      and calls[0]["id"] == "call_9"
      and calls[0]["arguments"] == {"expr": "2+2"})
check("chat_tools keeps raw arguments",
      calls[0]["arguments_raw"] == '{"expr": "2+2"}')
check("chat_tools usage", usage["total_tokens"] == 10)
check("chat_tools sends tools in body",
      srv.seen[0]["tools"][0]["function"]["name"] == "calc")
srv.shutdown()

bad_args = {
    "choices": [{"message": {"role": "assistant", "content": "x",
                 "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "calc",
                                              "arguments": "{not json"}}]}}],
    "usage": {},
}
srv = start_server([(200, bad_args)])
c = ChatClient(base_url="http://127.0.0.1:%d" % srv.server_port,
               api_key="x", max_retries=0)
_, _, calls = c.chat_tools([{"role": "user", "content": "hi"}], [CALC.schema()])
check("malformed arguments json becomes empty dict",
      calls[0]["arguments"] == {})
srv.shutdown()

plain = {"choices": [{"message": {"role": "assistant", "content": "yo"}}],
         "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                   "total_tokens": 2}}
srv = start_server([(200, plain)])
c = ChatClient(base_url="http://127.0.0.1:%d" % srv.server_port,
               api_key="x", max_retries=0)
text, usage, calls = c.chat_tools([{"role": "user", "content": "hi"}],
                                  [CALC.schema()])
check("chat_tools with no calls", text == "yo" and calls == [])
srv.shutdown()

# --- agent.act ---

plain_agent = Agent("maya", "a debater")
sc = ToolScriptedClient([("just talking", None)])
text, usage, turns = plain_agent.act("go", sc)
check("act without tools is one call",
      text == "just talking" and sc.chat_calls == 1
      and sc.tools_calls == 0 and turns == []
      and usage["total_tokens"] == 15)

calc_agent = Agent("maya", "a debater", tools=[CALC])
sc = ToolScriptedClient([
    ("", calc_call("2+2")),
    ("ready to answer", []),
    ("the answer is 4", None),
])
text, usage, turns = calc_agent.act("what is 2+2?", sc)
check("act runs the tool and answers",
      text == "the answer is 4" and len(turns) == 1)
check("tool turn kind and content",
      turns[0].kind == "tool: calc" and "4" in turns[0].text
      and "calc" in turns[0].text)
check("act merges usage across the turn",
      usage["total_tokens"] == 45 and calc_agent.tokens_used == 45)
check("tool schemas sent to the model", sc.sent_tools[0] == ["calc"])
check("final answer call has no tools", sc.chat_calls == 1)

# second chat_tools call gets the tool result message
msgs = sc.script  # exhausted; re-run to inspect messages
sc2 = ToolScriptedClient([
    ("", calc_call("3*3")),
    ("done", []),
    ("9", None),
])
seen = {}


class Spy(ToolScriptedClient):
    def chat_tools(self, messages, tools, model=None, temperature=None,
                   max_tokens=None):
        seen.setdefault("n", 0)
        seen["n"] += 1
        if seen["n"] == 2:
            seen["messages"] = messages
        return super().chat_tools(messages, tools, model=model,
                                  temperature=temperature,
                                  max_tokens=max_tokens)


Agent("w", "r", tools=[CALC]).act("go", Spy([("", calc_call("3*3")),
                                             ("done", []), ("9", None)]))
roles = [m["role"] for m in seen["messages"]]
check("assistant tool_calls message precedes tool results",
      roles == ["system", "user", "assistant", "tool"])
check("tool result message carries the call id",
      seen["messages"][3]["tool_call_id"] == "call_1"
      and seen["messages"][3]["content"] == "9")

err_agent = Agent("e", "r", tools=[Tool("b", "d", {}, boom)])
sc = ToolScriptedClient([("", [{"id": "c1", "name": "b", "arguments": {},
                                "arguments_raw": "{}"}]),
                         ("recovered", []), ("ok anyway", None)])
text, _, turns = err_agent.act("go", sc)
check("tool failure feeds back as error text",
      "error: ValueError: kaput" in turns[0].text and text == "ok anyway")

unk_agent = Agent("u", "r", tools=[CALC])
sc = ToolScriptedClient([("", [{"id": "c2", "name": "frobnicate",
                                "arguments": {}, "arguments_raw": "{}"}]),
                         ("moving on", []), ("fine", None)])
text, _, turns = unk_agent.act("go", sc)
check("unknown tool name becomes error text, no crash",
      "no tool named 'frobnicate'" in turns[0].text and text == "fine")

loop_agent = Agent("loopy", "r", tools=[CALC], max_tool_steps=1)
sc = ToolScriptedClient([
    ("", calc_call("1+1", "c1")),
    ("", calc_call("2+2", "c2")),   # past the cap, must not execute
    ("i give up and answer", None),
])
text, _, turns = loop_agent.act("go", sc)
check("max_tool_steps caps the loop",
      sc.tools_calls == 2 and sc.chat_calls == 1
      and len(turns) == 1 and text == "i give up and answer")

tight = Agent("tight", "r", tools=[CALC], budget_tokens=30)
sc = ToolScriptedClient([
    ("", calc_call("1+1", "c1")),
    ("", calc_call("2+2", "c2")),
    ("answer", None),
])
try:
    tight.act("go", sc)
    check("tool rounds respect the budget", False)
except BudgetExceeded:
    check("tool rounds respect the budget", True)
check("budget counts tool-round tokens", tight.tokens_used == 30)
check("over-budget final call never sent", sc.chat_calls == 0)

# --- patterns run tool-using agents ---

maya = Agent("maya", "argue for", tools=[CALC])
theo = Agent("theo", "argue against")


def debate_script():
    return [
        ("", calc_call("40+2", "m1")),   # maya tool round 1
        ("maya point soon", []),         # maya tool round 2, no calls
        ("maya makes her point", None),  # maya final answer
        ("theo makes his point", None),  # theo, no tools
        ("verdict: maya wins", None),    # judge
    ]


res = debate(ToolScriptedClient(debate_script()), "q?", [maya, theo], rounds=1)
tool_turns = [t for t in res.turns if t.kind == "tool: calc"]
check("debate records tool turns", len(tool_turns) == 1)
check("tool turn precedes the agent turn",
      res.turns.index(tool_turns[0]) < res.turns.index(
          next(t for t in res.turns if t.speaker == "maya"
               and t.kind == "round 1")))
check("debate answer still the verdict", "verdict" in res.answer)
check("debate usage includes tool rounds",
      res.usage_totals()["total_tokens"] == 5 * 15)


def sup_script():
    return [
        ("SUBTASK: do the math", None),          # supervisor plan
        ("", calc_call("6*7", "w1")),             # worker tool round
        ("partial", []),                          # worker, no more calls
        ("w1 did 42", None),                      # worker final
        ("merged: 42", None),                     # supervisor merge
    ]


boss = Agent("boss", "supervisor")
w1 = Agent("w1", "worker", tools=[CALC])
res = supervise(ToolScriptedClient(sup_script()), "do stuff", boss, [w1])
check("supervise worker tool turns recorded",
      any(t.kind == "tool: calc" for t in res.turns))
check("supervise still merges", res.turns[-1].kind == "merge")

# --- config ---

cfg_agent = Agent.from_dict({"name": "x", "tools": ["calc"],
                             "max_tool_steps": 5})
check("from_dict resolves tools",
      len(cfg_agent.tools) == 1 and cfg_agent.tools[0].name == "calc"
      and cfg_agent.max_tool_steps == 5)
try:
    Agent.from_dict({"name": "x", "tools": ["nope"]})
    check("from_dict rejects unknown tool", False)
except ValueError:
    check("from_dict rejects unknown tool", True)

# --- cli ---

import council.cli as cli_mod

ns = cli_mod.build_parser().parse_args(
    ["debate", "q?", "--agent", "a: r", "--tools", "calc", "--quiet"])
seen_agents = {}


def fake_debate(client, question, agents, rounds=3, judge=None, out=None):
    seen_agents["agents"] = agents
    from council.run import RunResult
    return RunResult("debate", question).finish("done")


old_debate = cli_mod.debate
cli_mod.debate = fake_debate
old_make = cli_mod.make_client
cli_mod.make_client = lambda args: ToolScriptedClient([])
try:
    cli_mod.cmd_debate(ns)
    got = seen_agents["agents"]
    check("cli --tools attaches to agents",
          len(got) == 1 and got[0].tools and got[0].tools[0].name == "calc")
except Exception as e:
    print("cli tools raised: %s" % e)
    check("cli --tools attaches to agents", False)
cli_mod.debate = old_debate
cli_mod.make_client = old_make

ns = cli_mod.build_parser().parse_args(
    ["debate", "q?", "--agent", "a: r", "--tools", "nope"])
cli_mod.make_client = lambda args: ToolScriptedClient([])
try:
    cli_mod.cmd_debate(ns)
    check("cli --tools rejects unknown tool", False)
except ValueError:
    check("cli --tools rejects unknown tool", True)
cli_mod.make_client = old_make

print("\n%d passed, %d failed" % (passed, failed))
sys.exit(1 if failed else 0)
