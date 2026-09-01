"""Activation access — capture (and later inject) residual-stream activations
during a training forward pass, with gradients intact.

``HookManager`` and ``pool`` have fixed signatures now so Phase 1 code can be
written against them; their bodies land in Phase 1 (capture / pooling) and
Phase 3 (inject).
"""

from interpost.activations.hooks import HookManager
from interpost.activations.pooling import pool

__all__ = ["HookManager", "pool"]
