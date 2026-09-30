"""an agent is just an identity wrapped around a chat client."""


class Agent:
    def __init__(self, name, role, system=None, model=None, temperature=0.7):
        self.name = name
        self.role = role
        self.system = system
        self.model = model
        self.temperature = temperature

    def system_prompt(self):
        base = "You are %s. %s" % (self.name, self.role)
        if self.system:
            base += "\n" + self.system
        return base

    def say(self, prompt, client):
        """one turn: system identity + the prompt, returns (text, usage)."""
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": prompt},
        ]
        return client.chat(messages, model=self.model,
                           temperature=self.temperature)

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d.get("role", "a helpful participant"),
                   system=d.get("system"), model=d.get("model"),
                   temperature=d.get("temperature", 0.7))

    def __repr__(self):
        return "Agent(%r)" % self.name
