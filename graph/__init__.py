"""C64-RE Agent — LangGraph package.

Public surface kept tiny on purpose:
    from graph import graph              # compiled StateGraph (for Studio)
    from graph.state import C64State     # shared state schema
"""

from graph.build import graph
from graph.state import C64State

__all__ = ["graph", "C64State"]
