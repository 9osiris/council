"""load a council from a json config file."""

import json

from .agent import Agent

PATTERNS = ("debate", "vote", "supervise", "pipeline", "fanout")


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    pattern = cfg.get("pattern")
    if pattern not in PATTERNS:
        raise ValueError("pattern must be one of %s, got %r"
                         % (", ".join(PATTERNS), pattern))
    agents = [Agent.from_dict(a) for a in cfg.get("agents", [])]
    judge = Agent.from_dict(cfg["judge"]) if cfg.get("judge") else None
    supervisor = (Agent.from_dict(cfg["supervisor"])
                  if cfg.get("supervisor") else None)
    reducer = (Agent.from_dict(cfg["reducer"])
               if cfg.get("reducer") else None)
    return {
        "pattern": pattern,
        "agents": agents,
        "judge": judge,
        "supervisor": supervisor,
        "reducer": reducer,
        "params": cfg.get("params", {}),
    }


def example_config():
    return {
        "pattern": "debate",
        "params": {
            "question": "should a coding agent ever rewrite tests to make them pass?",
            "rounds": 2,
        },
        "agents": [
            {"name": "maya", "role": "a pragmatic senior engineer who ships fast and trusts tests as contracts"},
            {"name": "theo", "role": "a careful systems thinker who distrusts shortcuts and values correctness proofs"},
        ],
        "judge": {"name": "moderator", "role": "a neutral moderator"},
    }
