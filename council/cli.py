"""command line interface for council."""

import argparse
import json
import sys

from . import __version__
from .agent import Agent
from .client import ChatClient, ApiError
from .config import example_config, load_config
from .patterns import debate, fanout, pipeline, supervise, vote
from .transcript import to_markdown


def parse_agent(spec):
    """'name: role description' -> Agent. role defaults to participant."""
    name, _, role = spec.partition(":")
    name = name.strip()
    role = role.strip() or "a helpful participant"
    if not name:
        raise ValueError("agent spec needs a name: %r" % spec)
    return Agent(name, role)


def add_common(p):
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--save", default=None,
                   help="write the full transcript as markdown")
    p.add_argument("--quiet", action="store_true",
                   help="print only the final answer")


def add_agents(p, flag="--agent", dest="agents"):
    p.add_argument(flag, dest=dest, action="append", default=[],
                   help="repeatable 'name: role description'")


def make_client(args):
    kw = {}
    if args.model:
        kw["model"] = args.model
    if args.base_url:
        kw["base_url"] = args.base_url
    if args.api_key:
        kw["api_key"] = args.api_key
    kw["timeout"] = args.timeout
    return ChatClient(**kw)


def printer(quiet):
    def show(speaker, text):
        if quiet:
            return
        print("\n[%s]\n%s" % (speaker, text.strip()))
    return show


def save_transcript(result, path):
    with open(path, "w") as f:
        f.write(to_markdown(result))
    print("saved transcript to %s" % path)


def cmd_init(args):
    cfg = example_config()
    with open("council.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print("wrote council.json")
    print("edit it, then run: council run council.json")


def cmd_run(args):
    cfg = load_config(args.config)
    client = make_client(args)
    out = printer(args.quiet)
    pattern = cfg["pattern"]
    agents = cfg["agents"]
    p = cfg["params"]
    if pattern == "debate":
        result = debate(client, p["question"], agents,
                        rounds=p.get("rounds", 3), judge=cfg["judge"], out=out)
    elif pattern == "vote":
        result = vote(client, p["topic"], p["options"], agents, out=out)
    elif pattern == "supervise":
        if not cfg["supervisor"]:
            sys.exit("supervise needs a supervisor in the config")
        result = supervise(client, p["task"], cfg["supervisor"], agents, out=out)
    elif pattern == "pipeline":
        result = pipeline(client, p["task"], agents, out=out)
    elif pattern == "fanout":
        result = fanout(client, p["task"], agents, reducer=cfg["reducer"], out=out)
    finish(result, args)


def finish(result, args):
    usage = result.usage_totals()
    if not args.quiet:
        print("\n== final answer ==\n%s" % result.answer.strip())
        print("\n(%d turns, %.1fs, %d tokens)" % (
            len(result.turns), result.seconds, usage["total_tokens"]))
    else:
        print(result.answer.strip())
    if args.save:
        save_transcript(result, args.save)


def cmd_debate(args):
    agents = [parse_agent(s) for s in args.agents] or [
        Agent("pro", "argue in favor, steelman the motion"),
        Agent("con", "argue against, steelman the opposition"),
    ]
    judge = parse_agent(args.judge) if args.judge else None
    result = debate(make_client(args), args.question, agents,
                    rounds=args.rounds, judge=judge, out=printer(args.quiet))
    finish(result, args)


def cmd_vote(args):
    agents = [parse_agent(s) for s in args.agents]
    if not agents:
        sys.exit("vote needs at least one --agent")
    if len(args.option) < 2:
        sys.exit("vote needs at least two --option")
    result = vote(make_client(args), args.topic, args.option, agents,
                  out=printer(args.quiet))
    finish(result, args)


def cmd_supervise(args):
    supervisor = parse_agent(args.supervisor)
    workers = [parse_agent(s) for s in args.worker]
    if not workers:
        sys.exit("supervise needs at least one --worker")
    result = supervise(make_client(args), args.task, supervisor, workers,
                       out=printer(args.quiet))
    finish(result, args)


def cmd_pipeline(args):
    stages = [parse_agent(s) for s in args.stage]
    if not stages:
        sys.exit("pipeline needs at least one --stage")
    result = pipeline(make_client(args), args.task, stages,
                      out=printer(args.quiet))
    finish(result, args)


def cmd_fanout(args):
    workers = [parse_agent(s) for s in args.worker]
    if not workers:
        sys.exit("fanout needs at least one --worker")
    reducer = parse_agent(args.reducer) if args.reducer else None
    result = fanout(make_client(args), args.task, workers, reducer=reducer,
                    out=printer(args.quiet))
    finish(result, args)


def build_parser():
    p = argparse.ArgumentParser(
        prog="council",
        description="multi-agent orchestration for openai-compatible apis")
    p.add_argument("--version", action="version", version="council " + __version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="scaffold an example council.json")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("run", help="run a council from a json config")
    s.add_argument("config")
    add_common(s)
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("debate", help="agents argue, judge decides")
    s.add_argument("question")
    add_agents(s)
    s.add_argument("--judge", default=None, help="'name: role'")
    s.add_argument("--rounds", type=int, default=3)
    add_common(s)
    s.set_defaults(func=cmd_debate)

    s = sub.add_parser("vote", help="agents vote on options")
    s.add_argument("topic")
    s.add_argument("--option", action="append", default=[])
    add_agents(s)
    add_common(s)
    s.set_defaults(func=cmd_vote)

    s = sub.add_parser("supervise", help="supervisor splits work, workers run it")
    s.add_argument("task")
    s.add_argument("--supervisor", required=True, help="'name: role'")
    s.add_argument("--worker", action="append", default=[])
    add_common(s)
    s.set_defaults(func=cmd_supervise)

    s = sub.add_parser("pipeline", help="stages transform output in order")
    s.add_argument("task")
    s.add_argument("--stage", action="append", default=[])
    add_common(s)
    s.set_defaults(func=cmd_pipeline)

    s = sub.add_parser("fanout", help="workers answer alone, reducer merges")
    s.add_argument("task")
    s.add_argument("--worker", action="append", default=[])
    s.add_argument("--reducer", default=None, help="'name: role'")
    add_common(s)
    s.set_defaults(func=cmd_fanout)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except ApiError as e:
        print("api error: %s" % e, file=sys.stderr)
        sys.exit(1)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        print("error: %s" % e, file=sys.stderr)
        sys.exit(2)
