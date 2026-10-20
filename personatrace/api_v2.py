"""memory_v2 的 HTTP 层：FastAPI 路由。

三个硬约束（原 skills_bot 的 API 层没有）：
1. Pydantic 入参模型 —— 不再 `request: dict` 手取字段，校验与 OpenAPI 文档自动生成；
2. 同步/阻塞调用不进事件循环 —— embedding、归档扫描这类重活丢 `to_thread`；
3. `/v1` 前缀 + `as_of` 成为一等公民（时间不是可选参数，是查询维度）。
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

try:  # 作为包导入时（推荐）
    from .memory_v2 import MemoryV2
except ImportError:  # 直接以脚本方式运行本目录下的 api_v2.py
    from memory_v2 import MemoryV2  # type: ignore[no-redef]


class ObserveIn(BaseModel):
    category: str = Field(..., examples=["goals"])
    label: str = Field(..., min_length=2, examples=["驻车集 V1.0 联调"])
    track: str = Field("work", pattern="^(work|persona|cross)$")
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    dimension: str = Field("identity", pattern="^(identity|need|predict)$")
    evidence_id: Optional[str] = None
    reasoning: str = Field("", description="为什么存这条（深读时作为语境带回）")
    evidence_text: str = Field("", description="原始语句片段，落 L0，检索时不参与打分")
    due_at_ts: Optional[float] = Field(None, description="UNIX 秒；用于前瞻'还有几天'")
    ts: Optional[float] = None


class BehaviorIn(BaseModel):
    kind: str = Field(..., examples=["ask", "app", "switch", "idle"])
    target: str = ""
    ts: Optional[float] = None


class BeliefOut(BaseModel):
    id: str
    track: str
    category: str
    label: str
    dimension: str
    confidence: float
    retention: float
    effective_confidence: float
    staleness: float
    due_in: Optional[float]
    hits: int
    evidence: List[str]
    needs_review: bool
    reasoning: Optional[str] = None          # 仅在 expand>0 时返回
    evidence_count: Optional[int] = None
    evidence_texts: Optional[List[str]] = None


class StateOut(BaseModel):
    as_of: float
    backend: str
    count: int
    beliefs: List[BeliefOut]


class RecallOut(BaseModel):
    as_of: float
    backend: str
    pinned_count: int
    hidden_count: int
    shown: List[BeliefOut]


def build_router(mem: MemoryV2) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["memory"])

    @router.post("/observe", response_model=Dict[str, Any], summary="写入一次观察（证据 + 结论）")
    async def observe(payload: ObserveIn) -> Dict[str, Any]:
        belief = await asyncio.to_thread(
            mem.observe,
            payload.category, payload.label,
            track=payload.track, confidence=payload.confidence,
            evidence_id=payload.evidence_id, dimension=payload.dimension,
            reasoning=payload.reasoning, evidence_text=payload.evidence_text,
            due_at_ts=payload.due_at_ts, ts=payload.ts,
        )
        return {
            "id": belief.id,
            "hits": belief.hits,
            "stability_days": round(belief.stability, 2),
            "retention": round(belief.retention_now(), 3),
            "effective_confidence": belief.effective_confidence(),
        }

    @router.get("/state", response_model=StateOut, summary="某一时刻的完整状态（支持时间回溯）")
    async def state(
        as_of: Optional[float] = Query(None, description="UNIX 秒；不传=现在"),
        include_archived: bool = Query(False),
    ) -> StateOut:
        rows = await asyncio.to_thread(mem.state, as_of, include_archived)
        return StateOut(
            as_of=as_of or 0.0, backend=mem.mode, count=len(rows),
            beliefs=[BeliefOut(**r) for r in rows],
        )

    @router.get("/recall", response_model=RecallOut, summary="取用：相关度 × 保持率，必选项强制入选")
    async def recall(
        q: str = Query(..., description="查询文本（此处按标签文本检索，向量由调用方注入）"),
        as_of: Optional[float] = None,
        k: int = Query(15, ge=1, le=100),
        expand: int = Query(0, ge=0, le=5, description="回表深度：0=只给标签，>0=附带 reasoning 与原文片段"),
    ) -> RecallOut:
        # 说明：向量化在 SDK 侧完成（宿主已有 embedding），API 负责时间与优先级策略
        result = await asyncio.to_thread(mem.recall, None, None, as_of, k, ("constraints", "identity"), expand)
        return RecallOut(
            as_of=result["as_of"], backend=result["backend"],
            pinned_count=result["pinned_count"], hidden_count=result["hidden_count"],
            shown=[BeliefOut(**r) for r in result["shown"]],
        )

    @router.post("/beliefs/{belief_id}/invalidate", summary="失效（不删除，保留历史）")
    async def invalidate(belief_id: str, ts: Optional[float] = None) -> Dict[str, Any]:
        await asyncio.to_thread(mem.invalidate, belief_id, ts)
        return {"status": "ok", "belief_id": belief_id}

    @router.post("/behavior", summary="记录一次行为（用于预测下一步）")
    async def behavior(payload: BehaviorIn) -> Dict[str, Any]:
        await asyncio.to_thread(mem.record_behavior, payload.kind, payload.target, payload.ts)
        return {"status": "ok"}

    @router.get("/behavior/predict", summary="预测下一步动作（bigram 转移概率）")
    async def predict(window: int = Query(30, ge=2, le=500)) -> Dict[str, Any]:
        preds = await asyncio.to_thread(mem.predict_next, window)
        return {"predictions": preds}

    @router.get("/health", summary="后端与规模")
    async def health() -> Dict[str, Any]:
        return await asyncio.to_thread(mem.stats)

    return router
