"""Per-delegation async pool for the slow external-provider tools."""

from __future__ import annotations

from .throttle import _cap_result


def _make_search_async_pool():
    """Factory → (asyncable, CollectSearches) sharing one fresh registry.

    External-provider calls (OpenAlex / Semantic Scholar / arXiv / PDF fetch)
    dominate literature wall-time — a single OpenAlex search can take minutes.
    `asyncable(provider, fn)` wraps a synchronous tool so the model can run it
    sync or async (the wait param, like Delegate's):
      • wait=True (DEFAULT): blocks and returns the result inline (the existing
        contract — a search returns results, not a handle).
      • wait=False: runs in a background thread, returns a handle immediately so
        the reviewer can fire independent searches on OTHER providers
        concurrently, then gather them with CollectSearches.
    Calls to the SAME provider serialize (one lock per provider — rate-limit
    safety); different providers run in parallel. `CollectSearches(handle?)`
    blocks until the named handle (or ALL pending) finish and returns results.
    A fresh pool is created per delegation (no cross-run state).
    """
    import inspect
    import itertools
    import threading

    prov_locks: dict = {}
    prov_guard = threading.Lock()
    reg: dict = {}
    reg_guard = threading.Lock()
    seq = itertools.count(1)

    def provider_lock(p):
        with prov_guard:
            return prov_locks.setdefault(p, threading.Lock())

    def asyncable(provider, fn):
        base_sig = inspect.signature(fn)
        wait_p = inspect.Parameter("wait", inspect.Parameter.KEYWORD_ONLY,
                                   default=True, annotation=bool)

        def wrapper(*args, **kwargs):
            _w = kwargs.pop("wait", True)
            # MCP string-in tools may pass "false"; coerce like Delegate.wait.
            wait = (_w if isinstance(_w, bool)
                    else str(_w).strip().lower() not in ("false", "0", "no", ""))
            if wait:
                with provider_lock(provider):
                    return _cap_result(fn(*args, **kwargs))
            h = f"{provider}#{next(seq)}"
            label = str((args[0] if args else None) or kwargs.get("query")
                        or kwargs.get("url") or kwargs.get("paper_id") or "")[:60]
            ev = threading.Event()
            rec = {"event": ev, "result": None, "label": label}
            with reg_guard:
                reg[h] = rec

            def run():
                try:
                    with provider_lock(provider):
                        rec["result"] = fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    rec["result"] = f"ERROR: {exc}"
                ev.set()

            threading.Thread(target=run, daemon=True, name=h).start()
            return (f"Started async on '{provider}' (handle {h}). Fire more "
                    "searches on OTHER providers now — they run concurrently — "
                    "then CollectSearches() to get all results before using them.")

        wrapper.__name__ = getattr(fn, "__name__", "tool")
        _doc = inspect.cleandoc(fn.__doc__ or "")
        wrapper.__doc__ = _doc + (
            "\n\nASYNC: pass wait=False to run this in the background and get a "
            "handle immediately, so you can fire independent searches on OTHER "
            "providers concurrently, then CollectSearches() for results; "
            "same-provider calls serialize. Default wait=True blocks and returns "
            "the result inline.")
        try:
            wrapper.__signature__ = base_sig.replace(
                parameters=list(base_sig.parameters.values()) + [wait_p])
            # inspect.signature() (parameter NAMES, via __signature__ above)
            # and typing.get_type_hints() (parameter TYPES, via
            # __annotations__ below) are two independent lookups -- pydantic's
            # schema builder uses both and KeyErrors if only one is set.
            wrapper.__annotations__ = {
                p.name: p.annotation for p in base_sig.parameters.values()
                if p.annotation is not inspect.Parameter.empty
            }
            wrapper.__annotations__["wait"] = bool
        except (ValueError, TypeError):
            pass
        return wrapper

    def CollectSearches(handle: str = None) -> str:
        """Collect async (wait=False) search results. With a handle: block until
        that search finishes and return its result. With NO handle: block until
        ALL pending async searches finish and return them all. Call this before
        using async search results — same-provider searches were serialized,
        different providers ran concurrently."""
        with reg_guard:
            handles = [handle] if handle else list(reg)
        if not handles:
            return "No async searches pending."
        parts = []
        for h in handles:
            rec = reg.get(h)
            if rec is None:
                parts.append(f"{h}: unknown or already-collected handle")
                continue
            done = rec["event"].wait(timeout=600)
            parts.append(f"=== {h} ({rec['label']}) ===\n" + (
                _cap_result(rec["result"]) if done else "(still running after 600s)"))
            with reg_guard:
                reg.pop(h, None)
        return "\n\n".join(parts)

    return asyncable, CollectSearches
