# council

multi-agent orchestration for any openai-compatible chat api. give a problem
to a group of agents instead of one and let them debate it, vote on it,
split it up, or pipeline it. stdlib only, no dependencies.

## patterns

- **debate** - agents argue for N rounds with the full transcript in context,
  then a judge writes the final verdict. good for open questions with no
  single right answer.
- **vote** - every agent picks one option with reasoning (`VOTE: <option>`),
  plurality wins. ties go to a runoff between the tied options. good for
  decisions.
- **supervise** - a supervisor breaks the task into subtasks, workers run
  them in parallel, supervisor merges. good for decomposable work.
- **pipeline** - stages run in order, each transforming the previous output.
  good for draft -> critique -> rewrite chains.
- **fanout** - every worker answers alone in parallel, a reducer merges the
  best of each. good for brainstorming then converging.

## install

```bash
pip install .
```

needs python 3.9+. talks to any openai-compatible endpoint:

```bash
export OPENAI_API_KEY=...
# or point it elsewhere
export OPENAI_BASE_URL=http://localhost:11400/v1
```

## quick start

```bash
# two agents debate, judge decides
council debate "should a coding agent ever rewrite tests to make them pass?" \
  --agent "maya: a pragmatic senior engineer who ships fast" \
  --agent "theo: a careful systems thinker who distrusts shortcuts" \
  --rounds 2

# vote on options
council vote "which router?" --option "round-robin" --option "least-loaded" \
  --agent "a1: an infra engineer" --agent "a2: an infra engineer" \
  --agent "a3: a skeptic"

# supervisor splits the work
council supervise "write a launch post for council" \
  --supervisor "boss: breaks work into subtasks and merges results" \
  --worker "w1: a technical writer" --worker "w2: a technical writer"

# pipeline: draft then edit then polish
council pipeline "explain raft consensus simply" \
  --stage "drafter: write a rough draft" \
  --stage "editor: tighten it, cut fluff" \
  --stage "polisher: final pass for clarity"

# fanout: three independent takes, merged
council fanout "name this project" \
  --worker "w1: likes short punchy names" \
  --worker "w2: likes descriptive names" \
  --worker "w3: likes weird names"
```

`--save transcript.md` writes the full run (every turn, the final answer,
token counts) as markdown. `--quiet` prints only the final answer.
`--model`, `--base-url`, `--api-key`, `--timeout` override the defaults.

## config files

for repeatable setups, `council init` scaffolds a `council.json`, then:

```bash
council run council.json --save run.md
```

```json
{
  "pattern": "debate",
  "params": {"question": "tabs or spaces?", "rounds": 3},
  "agents": [
    {"name": "maya", "role": "argue for tabs"},
    {"name": "theo", "role": "argue for spaces"}
  ],
  "judge": {"name": "moderator", "role": "a neutral moderator"}
}
```

vote configs use `params: {"topic": ..., "options": [...]}`;
supervise/pipeline/fanout use `params: {"task": ...}` and take
`supervisor` / `reducer` agent objects where relevant.

## budgets

agents can carry a token budget, and optional per-1k prices so cost is
tracked as tokens are spent. when a turn would push past the budget,
`say` raises `BudgetExceeded` instead of calling the api.

```python
from council import Agent, BudgetExceeded

scout = Agent("scout", "gather facts fast",
              budget_tokens=4000,
              prompt_price=0.15, completion_price=0.60)
try:
    scout.say("summarize the thread", client)
except BudgetExceeded as e:
    print(e)

print(scout.report())
# {'name': 'scout', 'tokens_used': 812, 'cost_used': 0.1722,
#  'budget_tokens': 4000}   (700 prompt + 112 completion tokens)
```

budgets also work from config files (`budget_tokens`, `prompt_price`,
`completion_price` per agent). per-agent totals show up in the saved
markdown under stats as `spend by agent`.

## library use

```python
from council import Agent, ChatClient, debate

client = ChatClient(model="gpt-4o-mini")
agents = [Agent("maya", "argue for"), Agent("theo", "argue against")]
result = debate(client, "tabs or spaces?", agents, rounds=2)
print(result.answer)
print(result.usage_totals())
```

there is also a `Blackboard`: a thread-safe shared scratchpad agents can
write to during a run (`write` / `append` / `read` / `dump_markdown`).
patterns don't force it on you; use it when agents need shared state.

## how the sausage is made

- every agent sees the system prompt `You are <name>. <role>` plus the task
  prompt. debate rounds render the whole transcript as `**name:** text`
  so agents actually respond to each other.
- votes are parsed from a `VOTE: <option>` line (case-insensitive), falling
  back to the first option mentioned in the text. no parse, no vote.
- supervise asks for `SUBTASK:` lines; if the supervisor rambles instead,
  it falls back to one generic subtask per worker so the run still works.
- retries 429/5xx with backoff. no streaming, no tools, no magic.

## tests

```bash
python tests/test_council.py
```

70 checks: a scripted fake client drives all five patterns (debate shape, vote tallies incl. ties and abstains, supervise split/merge, pipeline order, fanout reduce), budget enforcement (caps, pre-spent refusal, cost math, per-agent spend reporting), plus the real http client against a local fake openai server (response parsing, retry on 500, no retry on 400).

## license

MIT.
