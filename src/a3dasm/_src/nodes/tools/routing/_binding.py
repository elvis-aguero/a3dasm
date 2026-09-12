"""How a tool method is bound to one node without losing what describes it.

Tools are PascalCase methods on a per-node ``*Tools`` object, and three
separate consumers read them:

* ``prompts.tool_catalog`` renders ``__doc__`` as the model-facing description
* ``backends.claude._infer_schema_from_callable`` reads ``inspect.signature``
  to build the JSON schema — ``self`` is dropped, so a method's schema is
  exactly a plain function's
* ``viewer.readers._routing_tool_docs`` walks the module AST for PascalCase
  ``def``\\ s, which finds a method as readily as a nested closure

A plain bound method satisfies all three. The one thing it cannot do is carry
a docstring that differs per node, which two tools need — ``Delegate`` embeds
its node's connected targets, ``AskForFeedback`` the connected critic's
description. :func:`with_doc` is for exactly those.
"""
from __future__ import annotations

import functools
from collections.abc import Callable


def with_doc(fn: Callable, doc: str) -> Callable:
    """Re-present a bound tool method with a per-node docstring.

    ``update_wrapper`` carries across ``__name__`` and ``__dict__`` (hence any
    ``@tool_examples``), and the ``__wrapped__`` it sets keeps
    ``inspect.signature`` — and so the inferred JSON schema — identical to the
    method's own.
    """
    bound = functools.partial(fn)
    functools.update_wrapper(bound, fn)
    bound.__doc__ = doc
    return bound
