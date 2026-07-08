"""Agentic knowledge base (DRAFT).

The canonical source of truth for discretionary conventions and practices,
consultable on demand. See ``kb.py`` for the contract (what belongs here vs.
what is enforced elsewhere) and the roadmap to a worker-facing tool.
"""
from .kb import KBEntry, KnowledgeBase

__all__ = ["KBEntry", "KnowledgeBase"]
