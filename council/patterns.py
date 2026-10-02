"""the five ways a council of agents can work a problem."""

import re
import time
from concurrent.futures import ThreadPoolExecutor

from .agent import Agent
from .run import RunResult, Turn
from .transcript import render_transcript


def _timed(agent, prompt, client):
    start = time.time()
    text, usage, tool_turns = agent.act(prompt, client)
    return text, usage, tool_turns, time.time() - start


def _add(result, agent, text, usage, tool_turns, secs, kind, out=None):
    """tool turns land first (they happened mid-turn), then the main turn."""
    for t in tool_turns:
        result.add(t)
        if out:
            out(t.speaker, t.text)
    result.add(Turn(agent.name, text, usage, secs, kind=kind))
    if out:
        out(agent.name, text)


def _history_text(pairs):
    return render_transcript(pairs)


def debate(client, question, agents, rounds=3, judge=None, out=None):
    """agents argue for N rounds seeing the full transcript, judge decides."""
    result = RunResult("debate", question)
    pairs = [("question", question)]
    for r in range(rounds):
        for agent in agents:
            prompt = (
                "Debate transcript so far:\n\n%s\n\n"
                "Round %d of %d. Respond as %s: make your point, "
                "answer the others directly, stay under 150 words."
                % (_history_text(pairs), r + 1, rounds, agent.name))
            text, usage, tool_turns, secs = _timed(agent, prompt, client)
            pairs.append((agent.name, text))
            _add(result, agent, text, usage, tool_turns, secs,
                 "round %d" % (r + 1), out)
    judge = judge or Agent(
        "moderator",
        "a neutral moderator. read debates and write a clear final verdict.")
    prompt = (
        "Full debate transcript:\n\n%s\n\n"
        "Write the final verdict: which side won and why, in under 200 words. "
        "End with a one-line conclusion." % _history_text(pairs))
    text, usage, tool_turns, secs = _timed(judge, prompt, client)
    _add(result, judge, text, usage, tool_turns, secs, "verdict")
    return result.finish(text)


def _parse_vote(text, options):
    """find VOTE: <option>, else first option mentioned, else None."""
    m = re.search(r"^VOTE:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    if m:
        want = m.group(1).strip().lower()
        for opt in options:
            if opt.lower() == want or opt.lower() in want or want in opt.lower():
                return opt
    low = text.lower()
    for opt in options:
        if opt.lower() in low:
            return opt
    return None


def vote(client, topic, options, agents, out=None):
    """every agent votes with reasoning, plurality wins, ties go to runoff."""
    result = RunResult("vote", topic)
    tally = {opt: 0 for opt in options}
    ballots = []

    def one_round(candidates):
        votes = {}
        for agent in agents:
            prompt = (
                "Topic: %s\n\nOptions:\n%s\n\n"
                "Pick exactly one. Reply with a line 'VOTE: <option name>' "
                "followed by your reasoning in under 100 words."
                % (topic, "\n".join("- " + c for c in candidates)))
            text, usage, tool_turns, secs = _timed(agent, prompt, client)
            pick = _parse_vote(text, candidates)
            votes[agent.name] = (pick, text)
            _add(result, agent, text, usage, tool_turns, secs, "vote", out)
        return votes

    votes = one_round(options)
    for name, (pick, _) in votes.items():
        ballots.append((name, pick))
        if pick:
            tally[pick] += 1

    top = max(tally.values())
    winners = [o for o, n in tally.items() if n == top]
    if len(winners) > 1 and top > 0:
        # tie: runoff between the tied options only
        if out:
            out("moderator", "tie between %s, runoff" % ", ".join(winners))
        votes = one_round(winners)
        tally = {opt: 0 for opt in winners}
        for name, (pick, _) in votes.items():
            ballots.append((name, pick))
            if pick:
                tally[pick] += 1
        top = max(tally.values())
        winners = [o for o, n in tally.items() if n == top]

    winner = winners[0] if winners and top > 0 else None
    answer = "Winner: %s\nTally: %s" % (
        winner, ", ".join("%s=%d" % (o, tally[o]) for o in tally))
    result.extra = {"tally": tally, "winner": winner, "ballots": ballots}
    return result.finish(answer)


def _parse_subtasks(text):
    lines = re.findall(r"^SUBTASK:\s*(.+)$", text,
                       re.MULTILINE | re.IGNORECASE)
    return [l.strip() for l in lines if l.strip()]


def supervise(client, task, supervisor, workers, out=None, max_workers=4):
    """supervisor splits the task, workers run subtasks in parallel, merge."""
    result = RunResult("supervise", task)
    prompt = (
        "Task: %s\n\nBreak this into concrete subtasks a worker agent can do "
        "alone. Reply with one line per subtask starting with 'SUBTASK:'. "
        "Aim for %d subtasks." % (task, len(workers)))
    text, usage, tool_turns, secs = _timed(supervisor, prompt, client)
    _add(result, supervisor, text, usage, tool_turns, secs, "plan")
    subtasks = _parse_subtasks(text)
    if not subtasks:
        # supervisor rambled: fall back to one subtask per worker
        subtasks = ["handle part %d of: %s" % (i + 1, task)
                    for i in range(len(workers))]

    def do_one(pair):
        worker, sub = pair
        p = ("Overall task: %s\n\nYour subtask: %s\n\n"
             "Do it and report the result concretely, under 200 words."
             % (task, sub))
        t, u, tt, s = _timed(worker, p, client)
        return worker, sub, t, u, tt, s

    pairs = [(workers[i % len(workers)], sub)
             for i, sub in enumerate(subtasks)]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        done = list(pool.map(do_one, pairs))
    for worker, sub, t, u, tt, s in done:
        _add(result, worker, t, u, tt, s, "subtask: " + sub[:60], out)

    joined = "\n\n".join("Worker %s did '%s':\n%s" % (w.name, s, t)
                         for w, s, t, _, _, _ in done)
    prompt = ("Original task: %s\n\nSubtask results:\n\n%s\n\n"
              "Merge these into one final answer, resolving conflicts, "
              "under 300 words." % (task, joined))
    text, usage, tool_turns, secs = _timed(supervisor, prompt, client)
    _add(result, supervisor, text, usage, tool_turns, secs, "merge")
    result.extra = {"subtasks": subtasks}
    return result.finish(text)


def pipeline(client, task, stages, out=None):
    """each stage transforms the previous stage's output, in order."""
    result = RunResult("pipeline", task)
    current = task
    for i, stage in enumerate(stages):
        if i == 0:
            prompt = ("Task: %s\n\nYou are step 1 (%s). Do your part and "
                      "output only the result." % (task, stage.role))
        else:
            prompt = ("Original task: %s\n\nPrevious step output:\n%s\n\n"
                      "You are step %d (%s). Take the previous output, apply "
                      "your step, output only the result."
                      % (task, current, i + 1, stage.role))
        text, usage, tool_turns, secs = _timed(stage, prompt, client)
        _add(result, stage, text, usage, tool_turns, secs,
             "stage %d" % (i + 1), out)
        current = text
    return result.finish(current)


def fanout(client, task, workers, reducer=None, out=None, max_workers=8):
    """every worker answers alone in parallel, reducer merges the answers."""
    result = RunResult("fanout", task)

    def do_one(worker):
        prompt = ("Task: %s\n\nAnswer independently, under 200 words."
                  % task)
        t, u, tt, s = _timed(worker, prompt, client)
        return worker, t, u, tt, s

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        done = list(pool.map(do_one, workers))
    for worker, t, u, tt, s in done:
        _add(result, worker, t, u, tt, s, "independent", out)

    reducer = reducer or Agent(
        "merger", "a neutral editor. merge independent drafts into one "
                  "answer, keeping the best of each.")
    joined = "\n\n".join("**%s:**\n%s" % (w.name, t)
                         for w, t, _, _, _ in done)
    prompt = ("Task: %s\n\nIndependent answers:\n\n%s\n\n"
              "Merge into one final answer, keeping the strongest points "
              "of each, under 300 words." % (task, joined))
    text, usage, tool_turns, secs = _timed(reducer, prompt, client)
    _add(result, reducer, text, usage, tool_turns, secs, "merge")
    return result.finish(text)
