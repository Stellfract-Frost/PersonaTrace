"""personatrace-doctor：三秒钟证明「记忆真的在跑」。

记忆类项目最大的流失不是"记不准"，而是"看不出它在工作"：
装完、聊几句、什么也没发生，于是被判定没用。这个自检把三件容易出问题的事
一次跑给你看：后端选对了吗、写得进去吗、重启还在吗。

用法：
    personatrace-doctor              # 只检查，不删东西
    personatrace-doctor --gc         # 顺手回收「归档且只提过一次」的条目
    PERSONA_DB=D:/tmp/mem.db personatrace-doctor

退出码：0 = 全通；1 = 有环节失败（会指出是哪一步）。
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

PROBE_USER = "__doctor__"
PROBE_LABEL = "doctor 自检：这条应当能读回来"


def _line(name: str, ok: bool, detail: str = "") -> None:
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {name}{('  ' + detail) if detail else ''}")


def main(argv: Optional[List[str]] = None) -> int:
    from .memory_v2 import MemoryV2, build_backend

    argv = list(sys.argv[1:] if argv is None else argv)
    keep = "--keep" in argv
    re_gc = "--gc" in argv

    print("PersonaTrace doctor")
    backend, mode = build_backend()
    path = getattr(backend, "path", None)
    print(f"  后端: {mode}" + (f" ({path})" if path else ""))
    degraded = False
    if mode == "memory":
        from .memory_v2 import last_backend_warning

        warning = last_backend_warning()
        if warning:
            degraded = True
            print(f"  警告: {warning}")
        print("  注意: 进程内后端，重启会丢（如需落盘请去掉 PERSONA_DB=off）")

    mem = MemoryV2(backend=backend, user_id=PROBE_USER)
    ok = not degraded

    try:
        mem.purge()
    except Exception:
        pass

    # 1) 写入
    try:
        belief = mem.observe("state", PROBE_LABEL, track="work", confidence=0.5, reasoning="自检")
        _line("写入", True, f"id={belief.id[:8]}")
    except Exception as exc:  # noqa: BLE001
        _line("写入", False, str(exc))
        ok = False

    # 2) 读回
    rows: List[Dict[str, Any]] = []
    try:
        rows = mem.state()
        hit = any(r.get("label") == PROBE_LABEL for r in rows)
        _line("读回", hit, f"{len(rows)} 条")
        ok = ok and hit
    except Exception as exc:  # noqa: BLE001
        _line("读回", False, str(exc))
        ok = False

    # 3) 重启后还在吗（SQLite 才做得到真跨进程；其余模式退化成本进程检查）
    try:
        if mode == "sqlite":
            from .memory_v2 import MemoryV2 as _M, SQLiteBackend

            fresh = SQLiteBackend(str(path))
            try:
                again = _M(backend=fresh, user_id=PROBE_USER).state()
            finally:
                fresh.close()
            survived = any(r.get("label") == PROBE_LABEL for r in again)
            _line("重启后读回", survived, f"{len(again)} 条（新连接）")
            ok = ok and survived
        else:
            _line("重启后读回", True, "跳过（非 SQLite 后端）")
    except Exception as exc:  # noqa: BLE001
        _line("重启后读回", False, str(exc))
        ok = False

    # 4) 归档与可回收
    try:
        ov = mem.overview()
        plan = mem.gc(dry_run=True)
        _line("归档情况", True,
              f"活跃 {ov.get('live', 0)} / 归档 {ov.get('archived', 0)}，"
              f"可回收 {plan.get('removed', 0)} 条")
        if re_gc:
            freed = mem.gc()
            _line("回收", True, f"删除 {freed.get('removed', 0)} 条、"
                               f"{freed.get('evidence', 0)} 条原文")
    except Exception as exc:  # noqa: BLE001
        _line("归档情况", False, str(exc))

    # 5) 清理
    if not keep:
        try:
            mem.purge()
            _line("清理自检数据", True)
        except Exception as exc:  # noqa: BLE001
            _line("清理自检数据", False, str(exc))

    print("  结论: " + ("记忆链路正常 —— 宿主只要每轮读一次、写一次就能生效"
                       if ok else "有环节失败，先按上面那一行排查"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
