"""council: multi-agent orchestration for openai-compatible apis."""

from .agent import Agent
from .blackboard import Blackboard
from .client import ChatClient
from .patterns import debate, fanout, pipeline, supervise, vote
from .run import RunResult, Turn

__version__ = "1.0.0"
__all__ = ["Agent", "Blackboard", "ChatClient", "RunResult", "Turn",
           "debate", "vote", "supervise", "pipeline", "fanout",
           "__version__"]
