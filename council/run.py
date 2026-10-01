"""result types shared by every orchestration pattern."""

import time


class Turn:
    def __init__(self, speaker, text, usage=None, seconds=0.0, kind="say"):
        self.speaker = speaker
        self.text = text
        self.usage = usage or {}
        self.seconds = seconds
        self.kind = kind  # say, verdict, vote, subtask, merge

    def __repr__(self):
        return "Turn(%r, %r...)" % (self.speaker, self.text[:30])


class RunResult:
    def __init__(self, pattern, task):
        self.pattern = pattern
        self.task = task
        self.turns = []
        self.answer = ""
        self.extra = {}
        self.seconds = 0.0
        self._started = time.time()

    def add(self, turn):
        self.turns.append(turn)

    def finish(self, answer):
        self.answer = answer
        self.seconds = time.time() - self._started
        return self

    def usage_totals(self):
        total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for t in self.turns:
            for k in total:
                total[k] += t.usage.get(k, 0)
        return total

    def spend_by_agent(self):
        """total tokens per speaker, in first-seen order."""
        per = {}
        for t in self.turns:
            row = per.setdefault(t.speaker, {
                "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0})
            for k in row:
                row[k] += t.usage.get(k, 0)
        return per


def merge_usage(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out
