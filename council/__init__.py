"""council: multi-agent orchestration for openai-compatible apis."""

from .agent import Agent, BudgetExceeded
from .blackboard import Blackboard
from .client import ChatClient
from .patterns import debate, fanout, pipeline, supervise, vote
from .run import RunResult, Turn
from .tools import Tool, blackboard_tools, register

__version__ = "1.2.0"
__all__ = ["Agent", "Blackboard", "BudgetExceeded", "ChatClient",
           "RunResult", "Turn", "Tool", "blackboard_tools", "register",
           "debate", "vote", "supervise", "pipeline",
           "fanout", "__version__"]
