"""local tools an agent can call mid-turn. stdlib only, no network."""

import ast
import json
import operator


class ToolError(Exception):
    """raise this from a tool func to send a clean error back to the model."""


class Tool:
    """a named function the model can request through openai-style tool calls.

    func takes the declared arguments as keyword args and returns a string
    (or a json-able value). it runs in-process, synchronously."""

    def __init__(self, name, description, parameters, func):
        self.name = name
        self.description = description
        self.parameters = parameters  # json schema for the arguments object
        self.func = func

    def schema(self):
        """the openai function-calling spec for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, arguments):
        """execute the tool, returning a string for the model.

        bad arguments and tool failures come back as 'error: ...' text so
        the run keeps going instead of crashing."""
        if not isinstance(arguments, dict):
            return "error: arguments must be an object, got %r" % arguments
        try:
            result = self.func(**arguments)
        except ToolError as e:
            return "error: %s" % e
        except TypeError as e:
            return "error: bad arguments: %s" % e
        except Exception as e:
            return "error: %s: %s" % (type(e).__name__, e)
        if result is None:
            return "ok"
        if isinstance(result, str):
            return result
        return json.dumps(result)


_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    # arithmetic only: numbers and +-*/%//** . never evals code.
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ToolError("only plain arithmetic is allowed")


def _calc(expr):
    return str(_safe_eval(ast.parse(expr, mode="eval")))


CALC = Tool(
    "calc",
    "evaluate a plain arithmetic expression. no variables, no function calls",
    {"type": "object",
     "properties": {"expr": {"type": "string",
                             "description": "e.g. (12 + 34) * 2"}},
     "required": ["expr"]},
    lambda expr: _calc(expr))


def blackboard_tools(bb, author=""):
    """read/write tools bound to a shared Blackboard instance.

    config files can only name instance-free tools like calc; use these
    from python when a run shares a blackboard."""
    def read(key):
        val = bb.read(key)
        if val is None:
            return "no value stored under %r" % key
        if isinstance(val, str):
            return val
        return json.dumps(val)

    def write(key, value):
        bb.write(key, value, author=author or "agent")
        return "wrote %r" % key

    return [
        Tool("blackboard_read",
             "read a value from the shared blackboard",
             {"type": "object",
              "properties": {"key": {"type": "string"}},
              "required": ["key"]},
             read),
        Tool("blackboard_write",
             "write a string value to the shared blackboard",
             {"type": "object",
              "properties": {"key": {"type": "string"},
                             "value": {"type": "string"}},
              "required": ["key", "value"]},
             write),
    ]


_REGISTRY = {}


def register(tool):
    """add a Tool to the registry so configs and --tools can name it."""
    _REGISTRY[tool.name] = tool
    return tool


def get(name):
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError("unknown tool %r (registered: %s)"
                         % (name, ", ".join(sorted(_REGISTRY)) or "none"))


def resolve(names):
    """['calc'] -> [Tool]. accepts a single name as a plain string too."""
    if isinstance(names, str):
        names = [names]
    return [get(n) for n in names]


register(CALC)
