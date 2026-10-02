"""an agent is just an identity wrapped around a chat client."""

import json

from .run import Turn, merge_usage
from .tools import resolve as resolve_tools


class BudgetExceeded(Exception):
    """raised when an agent would spend past its token budget."""


class Agent:
    def __init__(self, name, role, system=None, model=None, temperature=0.7,
                 budget_tokens=None, prompt_price=0.0, completion_price=0.0,
                 tools=None, max_tool_steps=3):
        self.name = name
        self.role = role
        self.system = system
        self.model = model
        self.temperature = temperature
        # optional cap on total tokens this agent may spend in a run
        self.budget_tokens = budget_tokens
        # usd per 1k tokens, used to track cost as tokens are spent
        self.prompt_price = prompt_price
        self.completion_price = completion_price
        # local tools the model may call mid-turn (list of Tool)
        self.tools = list(tools) if tools else []
        # how many tool-call rounds before the model must just answer
        self.max_tool_steps = max_tool_steps
        self.tokens_used = 0
        self.cost_used = 0.0

    def budget_left(self):
        if self.budget_tokens is None:
            return None
        return self.budget_tokens - self.tokens_used

    def spend(self, usage):
        """record usage from one turn, enforcing the token budget."""
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        self.tokens_used += pt + ct
        self.cost_used += pt / 1000 * self.prompt_price
        self.cost_used += ct / 1000 * self.completion_price
        if (self.budget_tokens is not None
                and self.tokens_used > self.budget_tokens):
            raise BudgetExceeded(
                "%s spent %d tokens, over budget of %d"
                % (self.name, self.tokens_used, self.budget_tokens))

    def reset_budget(self):
        self.tokens_used = 0
        self.cost_used = 0.0

    def report(self):
        return {"name": self.name, "tokens_used": self.tokens_used,
                "cost_used": round(self.cost_used, 6),
                "budget_tokens": self.budget_tokens}

    def system_prompt(self):
        base = "You are %s. %s" % (self.name, self.role)
        if self.system:
            base += "\n" + self.system
        return base

    def _check_budget(self):
        if (self.budget_tokens is not None
                and self.tokens_used >= self.budget_tokens):
            raise BudgetExceeded(
                "%s already spent %d of %d budgeted tokens"
                % (self.name, self.tokens_used, self.budget_tokens))

    def say(self, prompt, client):
        """one turn: system identity + the prompt, returns (text, usage)."""
        self._check_budget()
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": prompt},
        ]
        text, usage = client.chat(messages, model=self.model,
                                  temperature=self.temperature)
        self.spend(usage)
        return text, usage

    def act(self, prompt, client, tools=None, max_tool_steps=None):
        """a turn where the model may call tools.

        with no tools this is exactly say(). with tools, requested calls
        run locally and their results go back to the model until it
        answers or max_tool_steps rounds pass. returns
        (text, usage, tool_turns); usage merges every api call in the turn
        and tool_turns are Turn objects (kind "tool: <name>") for the
        transcript."""
        tools = self.tools if tools is None else tools
        if max_tool_steps is None:
            max_tool_steps = self.max_tool_steps
        if not tools:
            text, usage = self.say(prompt, client)
            return text, usage, []

        by_name = {t.name: t for t in tools}
        schemas = [t.schema() for t in tools]
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": prompt},
        ]
        tool_turns = []
        total = {}
        rounds = 0
        while True:
            self._check_budget()
            text, usage, calls = client.chat_tools(
                messages, schemas, model=self.model,
                temperature=self.temperature)
            self.spend(usage)
            total = merge_usage(total, usage)
            if not calls or rounds >= max_tool_steps:
                break
            rounds += 1
            assistant_calls = []
            for call in calls:
                tool = by_name.get(call["name"])
                if tool is None:
                    result = "error: no tool named %r" % call["name"]
                else:
                    result = tool.run(call["arguments"])
                tool_turns.append(Turn(
                    self.name,
                    "%s(%s) -> %s" % (call["name"],
                                     json.dumps(call["arguments"]),
                                     result),
                    {}, 0.0, kind="tool: " + call["name"]))
                assistant_calls.append({
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"],
                                 "arguments": call["arguments_raw"]},
                })
                messages.append({"role": "tool",
                                 "tool_call_id": call["id"],
                                 "content": result})
            messages.insert(-len(calls), {
                "role": "assistant", "content": text or "",
                "tool_calls": assistant_calls})
        if rounds:
            # one last plain call so the model answers with the results
            self._check_budget()
            text, usage = client.chat(messages, model=self.model,
                                      temperature=self.temperature)
            self.spend(usage)
            total = merge_usage(total, usage)
        return text, total, tool_turns

    @classmethod
    def from_dict(cls, d):
        tools = resolve_tools(d["tools"]) if d.get("tools") else []
        return cls(d["name"], d.get("role", "a helpful participant"),
                   system=d.get("system"), model=d.get("model"),
                   temperature=d.get("temperature", 0.7),
                   budget_tokens=d.get("budget_tokens"),
                   prompt_price=d.get("prompt_price", 0.0),
                   completion_price=d.get("completion_price", 0.0),
                   tools=tools,
                   max_tool_steps=d.get("max_tool_steps", 3))

    def __repr__(self):
        return "Agent(%r)" % self.name
