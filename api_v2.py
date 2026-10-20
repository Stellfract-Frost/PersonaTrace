"""兼容入口：`from api_v2 import build_router`。

实现在 `personatrace/api_v2.py`，用法不变：
    from api_v2 import build_router
    app.include_router(build_router(MemoryV2()))
"""

from personatrace.api_v2 import BehaviorIn, BeliefOut, ObserveIn, RecallOut, StateOut, build_router

__all__ = ["build_router", "ObserveIn", "BehaviorIn", "BeliefOut", "StateOut", "RecallOut"]
