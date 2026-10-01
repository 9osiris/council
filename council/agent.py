"""an agent is just an identity wrapped around a chat client."""


class BudgetExceeded(Exception):
    """raised when an agent would spend past its token budget."""


class Agent:
    def __init__(self, name, role, system=None, model=None, temperature=0.7,
                 budget_tokens=None, prompt_price=0.0, completion_price=0.0):
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

    def say(self, prompt, client):
        """one turn: system identity + the prompt, returns (text, usage)."""
        if (self.budget_tokens is not None
                and self.tokens_used >= self.budget_tokens):
            raise BudgetExceeded(
                "%s already spent %d of %d budgeted tokens"
                % (self.name, self.tokens_used, self.budget_tokens))
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": prompt},
        ]
        text, usage = client.chat(messages, model=self.model,
                                  temperature=self.temperature)
        self.spend(usage)
        return text, usage

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d.get("role", "a helpful participant"),
                   system=d.get("system"), model=d.get("model"),
                   temperature=d.get("temperature", 0.7),
                   budget_tokens=d.get("budget_tokens"),
                   prompt_price=d.get("prompt_price", 0.0),
                   completion_price=d.get("completion_price", 0.0))

    def __repr__(self):
        return "Agent(%r)" % self.name
