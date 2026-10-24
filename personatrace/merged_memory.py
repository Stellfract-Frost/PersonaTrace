"""Merged memory facade: v1 retrieval + v2 beliefs + short-term intentions.

A belief is a long-lived proposition; an intention is a one-off commitment with
a lifecycle. They share one prompt but not one table:

    MemoryV2     beliefs      retention / as_of / evidence
    IntentStore  intentions   importance / due / status / grace

The split is deliberate. A one-off "后天要上课" must not be merged into a generic
"上课" belief by label, and must not live for 45 days just because it landed in
``goals``. Conversely, a long-term preference must not disappear one day after
a deadline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .episodes import EpisodeStore, has_retrospective
from .facet import has_time_intent, humanize_due, render_time_line, time_bucket
from .intentions import IntentStore
from .routines import RoutineStore
from .recall import rank
from .schema import (
    ALL_CATEGORIES, DEFAULT_TZ, STABILITY_DAYS, BeliefIn, coerce_in,
    due_for_write, parse_due, track_for,
)
from .structured import call_structured

logger = logging.getLogger(__name__)

WRITE_SOURCES = frozenset({"direct", "bridge", "console"})
DEFAULT_TTL_S = 86400
DEFAULT_LEAD_S = 2 * 86400


def _zone(tz: str):
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(tz or DEFAULT_TZ)
    except Exception:
        from zoneinfo import ZoneInfo
        return ZoneInfo(DEFAULT_TZ)


def _aware(now: Optional[datetime], tz: str) -> datetime:
    now = now or datetime.now(_zone(tz))
    if now.tzinfo is None:
        return now.replace(tzinfo=_zone(tz))
    return now.astimezone(_zone(tz))


# Conservative write-time dedup. Exact labels are already idempotent in
# MemoryV2; this catches only near-identical paraphrases. It is lexical on
# purpose: it must work without an embedding model and it must not collapse
# two different commitments that happen to share a verb.
_DEDUP_THRESHOLD = 0.86
_DEDUP_SYNONYMS = (
    ("偏好", "喜欢"),
    ("比较", ""),
    ("很", ""),
    ("正在", "在"),
    ("在赶", "在做"),
)


def _normalize_label(label: str) -> str:
    text = (label or "").strip().lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"^[我的]+", "", text)
    for old, new in _DEDUP_SYNONYMS:
        text = text.replace(old, new)
    return text


def _label_similarity(a: str, b: str) -> float:
    left, right = _normalize_label(a), _normalize_label(b)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def _time_compatible(
    a_due: Optional[float],
    a_kind: str,
    b_due: Optional[float],
    b_kind: str,
) -> bool:
    """A paraphrase may merge only when it points at the same time slot."""
    a_kind = a_kind or "none"
    b_kind = b_kind or "none"
    if a_kind != b_kind:
        return False
    if a_kind == "none":
        return True
    if a_due is None or b_due is None:
        return False
    return abs(float(a_due) - float(b_due)) <= 86400.0


def _clamp(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number > 1.0:
        number = number / 5.0 if number <= 5.0 else (number / 100.0 if number <= 100.0 else 1.0)
    return max(0.0, min(1.0, number))


class IntentIn(BaseModel):
    """One short-term intention as extracted from a turn."""

    model_config = ConfigDict(extra="forbid")

    action: str = Field(min_length=2, max_length=120)
    detail: str = Field(default="", max_length=300)
    due: Optional[str] = Field(default=None, max_length=80)
    ttl_hours: Optional[float] = Field(default=None, gt=0, le=24 * 30)
    importance: float = Field(default=0.6, ge=0.0, le=5.0)
    confidence: float = Field(default=0.6, ge=0.0, le=5.0)
    needs_confirmation: bool = False
    recurrence: Optional[str] = Field(default=None, max_length=200)


class MemoryExtraction(BaseModel):
    """One LLM write response: long-term beliefs + short-term intentions."""

    model_config = ConfigDict(extra="forbid")
    beliefs: List[BeliefIn] = Field(default_factory=list)
    intentions: List[IntentIn] = Field(default_factory=list)


@dataclass
class ContextPack:
    """Structured read result. Rendering is a separate, optional adapter."""

    now: datetime
    query: str
    pinned: List[Dict[str, Any]] = field(default_factory=list)
    dated: List[Dict[str, Any]] = field(default_factory=list)
    related: List[Dict[str, Any]] = field(default_factory=list)
    cross: List[Dict[str, Any]] = field(default_factory=list)
    due_review: List[Dict[str, Any]] = field(default_factory=list)
    agenda_lines: List[str] = field(default_factory=list)
    episodes: List[Dict[str, Any]] = field(default_factory=list)
    routine_probes: List[Dict[str, Any]] = field(default_factory=list)
    hidden_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "now": self.now.isoformat(),
            "query": self.query,
            "pinned": self.pinned,
            "dated": self.dated,
            "related": self.related,
            "cross": self.cross,
            "due_review": self.due_review,
            "agenda_lines": self.agenda_lines,
            "episodes": self.episodes,
            "routine_probes": self.routine_probes,
            "hidden_count": self.hidden_count,
        }


class MergedMemory:
    """Facade over MemoryV2 + IntentStore.

    ``remember`` writes both types; ``build_context`` returns structured data;
    ``inject`` is an optional prompt adapter. The host still owns tone, tool
    policy and when to call.
    """

    def __init__(
        self,
        memory: Any = None,
        *,
        llm: Any = None,
        embed_fn: Any = None,
        intents: Optional[IntentStore] = None,
        episodes: Optional[EpisodeStore] = None,
        routines: Optional[RoutineStore] = None,
        vector_index: Any = None,
        user_id: str = "default",
        redis_url: Optional[str] = None,
        sqlite_path: Optional[str] = None,
        intents_path: Optional[str] = None,
        episodes_path: Optional[str] = None,
        routines_path: Optional[str] = None,
        tz: str = DEFAULT_TZ,
        default_ttl_s: int = DEFAULT_TTL_S,
        write_sources: Iterable[str] = WRITE_SOURCES,
    ) -> None:
        self.user_id = user_id
        self.tz = tz or DEFAULT_TZ
        self.llm = llm
        self.embed_fn = embed_fn
        self.default_ttl_s = max(1, int(default_ttl_s))
        self.write_sources = frozenset(write_sources)
        self.memory = memory or self._build_memory(user_id, redis_url, sqlite_path)
        # If a local SQLite file is used, keep intentions in the same file by
        # default: one file for the whole memory, not one file per subsystem.
        resolved_intents_path = intents_path or (sqlite_path if memory is None else None) or ":memory:"
        self.intents = intents or IntentStore(resolved_intents_path, default_tz=self.tz)
        resolved_episodes_path = episodes_path or (sqlite_path if memory is None else None) or ":memory:"
        resolved_routines_path = routines_path or (sqlite_path if memory is None else None) or ":memory:"
        self.episodes = episodes or EpisodeStore(
            resolved_episodes_path, default_tz=self.tz, vector_index=vector_index,
        )
        self.routines = routines or RoutineStore(resolved_routines_path, default_tz=self.tz)
        self._vec_cache: Dict[str, Sequence[float]] = {}
        # The 17-category stability table is a runtime config, not a kernel
        # fork. Existing beliefs keep their old stability; new ones get this.
        from .memory_v2 import CATEGORY_STABILITY
        CATEGORY_STABILITY.update(STABILITY_DAYS)

    @staticmethod
    def _build_memory(user_id: str, redis_url: Optional[str], sqlite_path: Optional[str] = None):
        from .memory_v2 import MemoryV2, SQLiteBackend
        if sqlite_path:
            return MemoryV2(backend=SQLiteBackend(sqlite_path), user_id=user_id)
        # Remote kernel already defaults to SQLite, then Redis/memory fallback.
        return MemoryV2(user_id=user_id, db_path=sqlite_path)

    # ---------------------------------------------------------------- write
    def remember(
        self,
        user_msg: str,
        reply: str = "",
        *,
        source: str = "direct",
        now: Optional[datetime] = None,
        force: bool = False,
        session_id: str = "",
    ) -> Dict[str, Any]:
        text = (user_msg or "").strip()
        if not text:
            return {"status": "skipped", "reason": "empty_user_message"}
        if source not in self.write_sources:
            return {"status": "skipped", "reason": "source_not_allowed", "source": source}

        now_dt = _aware(now, self.tz)
        # Store the raw turn even if extraction is throttled or returns nothing:
        # the turn is provenance for later retrospective search.
        episode = self.episodes.add(
            self.user_id, text, reply,
            session_id=session_id, source=source,
            ts=now_dt.timestamp(),
        )
        episode_id = episode.id if episode else ""

        if not force:
            cold = int(self.memory.stats().get("beliefs", 0)) < 10
            if not self.memory.allow("merged_extract", 120 if cold else 600):
                return {"status": "skipped", "reason": "throttled", "episode_id": episode_id}

        extraction = self._extract(text, reply, now_dt)
        if extraction is None:
            return {"status": "no_result", "episode_id": episode_id}

        beliefs = self._write_beliefs(extraction.beliefs, text, now_dt, episode_id=episode_id)
        intents = self._write_intentions(extraction.intentions, text, now_dt)
        if episode_id and (beliefs or intents):
            self.episodes.link(
                episode_id,
                belief_ids=[item.id for item in beliefs],
                intention_ids=[item.id for item in intents],
            )
        return {
            "status": "success" if (beliefs or intents) else "no_change",
            "beliefs": len(beliefs),
            "intentions": len(intents),
            "episode_id": episode_id,
            "belief_ids": [item.id for item in beliefs],
            "intent_ids": [item.id for item in intents],
        }

    def record_turn(
        self,
        user_msg: str,
        reply: str = "",
        *,
        source: str = "direct",
        session_id: str = "",
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Store a raw turn only; extraction/decisions stay on the host side."""
        text = (user_msg or "").strip()
        if not text:
            return {"status": "skipped", "reason": "empty_user_message"}
        if source not in self.write_sources:
            return {"status": "skipped", "reason": "source_not_allowed", "source": source}
        episode = self.episodes.add(
            self.user_id, text, reply,
            session_id=session_id, source=source,
            ts=_aware(now, self.tz).timestamp(),
        )
        return {"status": "success" if episode else "skipped",
                "episode_id": episode.id if episode else ""}

    def write(
        self,
        beliefs: Iterable[Any] = (),
        intentions: Iterable[Any] = (),
        *,
        source: str = "direct",
        now: Optional[datetime] = None,
        episode_id: str = "",
    ) -> Dict[str, Any]:
        """Write already-structured items (MCP / host-side extraction).

        This does not call an LLM and does not require a user message; the
        caller is responsible for only passing items extracted from a user
        turn. The source whitelist still applies.
        """
        if source not in self.write_sources:
            return {"status": "skipped", "reason": "source_not_allowed", "source": source}
        now_dt = _aware(now, self.tz)

        valid_beliefs: List[BeliefIn] = []
        invalid_beliefs = 0
        for raw in beliefs or []:
            item = coerce_in(raw)
            if item is None:
                invalid_beliefs += 1
            else:
                valid_beliefs.append(item)

        valid_intents: List[IntentIn] = []
        invalid_intents = 0
        for raw in intentions or []:
            try:
                valid_intents.append(raw if isinstance(raw, IntentIn) else IntentIn.model_validate(raw))
            except Exception:
                invalid_intents += 1

        written_beliefs = self._write_beliefs(valid_beliefs, "", now_dt, episode_id=episode_id)
        written_intents = self._write_intentions(valid_intents, "", now_dt)
        if episode_id and (written_beliefs or written_intents):
            self.episodes.link(
                episode_id,
                belief_ids=[item.id for item in written_beliefs],
                intention_ids=[item.id for item in written_intents],
            )
        return {
            "episode_id": episode_id,
            "status": "success" if (written_beliefs or written_intents) else "no_change",
            "beliefs": len(written_beliefs),
            "intentions": len(written_intents),
            "invalid_beliefs": invalid_beliefs,
            "invalid_intentions": invalid_intents,
            "belief_ids": [item.id for item in written_beliefs],
            "intent_ids": [item.id for item in written_intents],
        }

    def agenda(self, *, now: Optional[datetime] = None, max_lines: int = 6) -> Dict[str, Any]:
        """Structured short-term agenda. No prompt wording, no side effects."""
        now_dt = _aware(now, self.tz)
        return self.intents.select_agenda(self.user_id, now=now_dt, max_lines=max_lines)

    def label_routine(
        self,
        action: str,
        *,
        time: str = "09:00",
        weekdays: Iterable[int] = (),
        interval: int = 1,
        anchor_date: Any = None,
        valid_from: Any = None,
        valid_until: Any = None,
        exdates: Iterable[Any] = (),
        rdates: Iterable[Any] = (),
        status: str = "candidate",
        confidence: float = 0.55,
        allow_probe: bool = True,
        sensitivity: str = "",
        note: str = "",
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Explicit user label for a weekly routine; no keyword action list."""
        routine = self.routines.label(
            self.user_id, action,
            time=time, weekdays=weekdays, interval=interval,
            anchor_date=anchor_date or _aware(now, self.tz).date(),
            timezone=self.tz, valid_from=valid_from, valid_until=valid_until,
            exdates=exdates, rdates=rdates, status=status,
            confidence=confidence, allow_probe=allow_probe,
            sensitivity=sensitivity, note=note,
        )
        return routine.to_dict() if routine else {"status": "skipped"}

    def observe_routine(
        self,
        action: str,
        *,
        occurrence_date: Any = None,
        time: Optional[str] = None,
        weekdays: Optional[Iterable[int]] = None,
        interval: int = 1,
        episode_id: str = "",
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        now_dt = _aware(now, self.tz)
        routine = self.routines.observe(
            self.user_id, action, occurred_at=now_dt,
            occurrence_date=occurrence_date, time=time, weekdays=weekdays,
            interval=interval, timezone=self.tz, episode_id=episode_id,
        )
        return routine.to_dict() if routine else {"status": "skipped"}

    def routine_probes(self, *, now: Optional[datetime] = None, limit: int = 3) -> List[Dict[str, Any]]:
        return self.routines.probe(self.user_id, now=_aware(now, self.tz), limit=limit)

    def routine_feedback(
        self,
        probe_id: str,
        result: str,
        *,
        now: Optional[datetime] = None,
        note: str = "",
    ) -> Dict[str, Any]:
        routine = self.routines.feedback(
            probe_id, result, now=_aware(now, self.tz), note=note,
        )
        return routine.to_dict() if routine else {"status": "skipped"}

    def episodes_search(
        self,
        query: str = "",
        *,
        after: Any = None,
        before: Any = None,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        return self.episodes.search(
            self.user_id, query, after=after, before=before, limit=limit,
        )

    def _extract(self, user_msg: str, reply: str, now: datetime) -> Optional[MemoryExtraction]:
        if self.llm is None:
            return None
        prompt = self._build_prompt(user_msg, reply, now)
        try:
            return call_structured(
                self.llm, prompt, MemoryExtraction, retries=2, max_tokens=1600
            )
        except Exception:
            logger.warning("memory extraction failed; turn skipped", exc_info=True)
            return None

    def _build_prompt(self, user_msg: str, reply: str, now: datetime) -> str:
        categories = ", ".join(ALL_CATEGORIES)
        return f"""You extract memory from one conversation turn.
Current local time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')} (timezone {self.tz})

Return TWO kinds of records:

1. beliefs: stable facts/preferences about the user, one of 17 categories:
   {categories}
   Beliefs have NO due date. Do not put a one-off task here.

2. intentions: one-off future actions/events/deadlines. Each intention may
   have a due expression. importance means consequence if missed, NOT how often
   it was mentioned. If there is no due date, set ttl_hours (default 24).
   Do not turn a recurring habit ("每天十点睡") into an intention.

Rules:
- Skip past events, third-party plans, hypotheticals, and assistant actions.
- Do not calculate dates. Return the user's surface expression in due.
- If the user only said it once but it is a commitment, keep it; frequency is
  irrelevant.
- If ambiguous, set needs_confirmation true rather than inventing details.
- Empty lists are valid.

User: {user_msg}
Assistant: {(reply or '')[:400]}

Return JSON only:
{{
  "beliefs": [
    {{"category": "goals", "label": "...", "confidence": 0.0,
      "reasoning": "...", "evidence_text": "..."}}
  ],
  "intentions": [
    {{"action": "...", "detail": "...", "due": "后天",
      "ttl_hours": null, "importance": 0.0, "confidence": 0.0,
      "needs_confirmation": false, "recurrence": null}}
  ]
}}"""

    def _existing_belief_index(self, now: datetime) -> Dict[tuple, List[Dict[str, Any]]]:
        index: Dict[tuple, List[Dict[str, Any]]] = {}
        for row in self.memory.state(as_of=now.timestamp(), expand=0):
            key = (row.get("track", "work"), row.get("category", ""))
            index.setdefault(key, []).append(row)
        return index

    def _find_dedup(
        self,
        index: Dict[tuple, List[Dict[str, Any]]],
        item: BeliefIn,
        track: str,
        due_ts: Optional[float],
        time_kind: str,
    ) -> Optional[Dict[str, Any]]:
        """Return an existing belief that is the same proposition, if any."""
        for row in index.get((track, item.category), []):
            if not _time_compatible(
                due_ts, time_kind,
                row.get("due_at_ts"), row.get("time_kind", "none"),
            ):
                continue
            if _label_similarity(item.label, row.get("label", "")) >= _DEDUP_THRESHOLD:
                return row
        return None

    def _write_beliefs(
        self,
        beliefs: List[BeliefIn],
        user_msg: str,
        now: datetime,
        *,
        episode_id: str = "",
    ) -> List[Any]:
        written = []
        index = self._existing_belief_index(now)
        for draft in beliefs[:4]:
            item = coerce_in(draft)
            if item is None:
                continue
            typed = due_for_write(item, now, self.tz)
            due_ts = typed.due_ts if typed else None
            time_kind = typed.kind if typed else "none"
            time_raw = typed.raw if typed else (item.due or "")
            track = item.track or track_for(item.category)
            match = self._find_dedup(index, item, track, due_ts, time_kind)
            label = match["label"] if match else item.label
            try:
                belief = self.memory.observe(
                    item.category, label,
                    track=track,
                    confidence=item.confidence,
                    reasoning=item.reasoning,
                    evidence_id=episode_id or None,
                    evidence_text="" if episode_id else (item.evidence_text or user_msg[:4000]),
                    due_at_ts=due_ts,
                    time_kind=time_kind,
                    time_raw=time_raw,
                    ts=now.timestamp(),
                )
                written.append(belief)
                if match is None:
                    index.setdefault((track, item.category), []).append({
                        "label": belief.label,
                        "track": belief.track,
                        "category": belief.category,
                        "due_at_ts": belief.due_at_ts,
                        "time_kind": belief.time_kind,
                    })
            except Exception:
                logger.warning("belief write failed: %s", item.label, exc_info=True)
        return written

    def _write_intentions(self, intentions: List[IntentIn], user_msg: str, now: datetime) -> List[Any]:
        written = []
        for draft in intentions[:4]:
            title = (draft.action or "").strip()
            if len(title) < 2:
                continue
            due = parse_due(draft.due, now, self.tz) if draft.due else None
            if due is not None and due.kind == "past":
                continue
            if due is not None and due.kind == "recurring":
                continue
            # A surface due we cannot parse must not silently become a 24h TTL.
            if draft.due and due is None and not draft.ttl_hours:
                continue

            if due is not None and due.due_ts is not None:
                due_at = due.due_ts
                granularity = _granularity(due.due_ts, self.tz)
                due_kind = _due_kind(title)
                grace_s = 86400 if due.kind != "range" else 0
                lead_s = _lead_seconds(title)
            else:
                ttl_s = int((draft.ttl_hours or 0) * 3600) if draft.ttl_hours else self.default_ttl_s
                due_at = now.timestamp() + max(1, ttl_s)
                granularity = "datetime"
                due_kind = "end"
                grace_s = 0                 # TTL task disappears at its TTL
                lead_s = min(DEFAULT_LEAD_S, max(3600, ttl_s))

            try:
                written.append(self.intents.upsert(
                    self.user_id, title,
                    due_at=due_at,
                    due_kind=due_kind,
                    tz=self.tz,
                    granularity=granularity,
                    kind="event" if due_kind == "start" else "task",
                    detail=draft.detail,
                    importance=_clamp(draft.importance),
                    confidence=_clamp(draft.confidence),
                    rrule=_rrule(draft.recurrence),
                    lead_s=lead_s,
                    grace_s=grace_s,
                    needs_confirmation=draft.needs_confirmation,
                    source_turn=now.isoformat(),
                    source_text=user_msg[:500],
                    now=now,
                ))
            except Exception:
                logger.warning("intention write failed: %s", title, exc_info=True)
        return written

    # ---------------------------------------------------------------- read
    def build_context(
        self,
        query: str,
        *,
        now: Optional[datetime] = None,
        k: int = 4,
        max_agenda_lines: int = 6,
        mark_agenda: bool = False,
        include_probes: bool = True,
    ) -> ContextPack:
        now_dt = _aware(now, self.tz)
        now_ts = now_dt.timestamp()
        rows = self.memory.state(as_of=now_ts, expand=1)
        query_vec, vectors = self._embed_for(query, rows)
        ranked = rank(
            rows, query, query_vec=query_vec, vectors=vectors, k=k,
        )
        dated = time_bucket(rows, now=now_dt, tz=self.tz) if has_time_intent(query) else []
        agenda_lines: List[str] = []
        try:
            agenda_text = self.intents.render_agenda(
                self.user_id, now=now_dt, max_lines=max_agenda_lines, commit=mark_agenda,
            )
            agenda_lines = [line for line in agenda_text.splitlines() if line.startswith("- ")]
        except Exception:
            logger.debug("agenda render failed", exc_info=True)

        dated_ids = {str(row.get("id")) for row in dated}
        due_review = []
        for row in rows:
            if str(row.get("id")) in dated_ids:
                continue
            if row.get("needs_review") or (row.get("due_in") is not None and row["due_in"] < 3):
                due_review.append(row)
        due_review = due_review[:3]

        episodes: List[Dict[str, Any]] = []
        if has_retrospective(query):
            try:
                episodes = self.episodes.search(self.user_id, query, limit=2)
            except Exception:
                logger.debug("episode search failed", exc_info=True)
        probes: List[Dict[str, Any]] = []
        if include_probes:
            try:
                probes = self.routines.probe(self.user_id, now=now_dt, limit=2)
            except Exception:
                logger.debug("routine probe failed", exc_info=True)

        return ContextPack(
            now=now_dt, query=query,
            pinned=ranked["pinned"], dated=dated, related=ranked["shown"],
            cross=ranked["cross"], due_review=due_review,
            agenda_lines=agenda_lines, episodes=episodes,
            routine_probes=probes, hidden_count=ranked["hidden_count"],
        )

    def inject(
        self,
        query: str,
        *,
        now: Optional[datetime] = None,
        k: int = 4,
        max_chars: int = 640,
        tool_hint: str = "memory_recall / memory_state",
    ) -> str:
        pack = self.build_context(query, now=now, k=k, mark_agenda=True)
        lines: List[str] = []

        if pack.pinned:
            lines.append("【底线/身份】")
            for row in pack.pinned:
                lines.append(_belief_line(row, pack.now, self.tz))
        if pack.dated:
            lines.append("【和日期有关的】")
            for row in pack.dated:
                line = _dated_line(row, pack.now, self.tz)
                if line:
                    lines.append(line)
        if pack.episodes:
            lines.append("【上次聊到】")
            for episode in pack.episodes:
                line = _episode_line(episode)
                if line:
                    lines.append(line)
        # Short-term tasks go before the long related tail: truncation must not
        # cut the one section whose absence is a commitment failure.
        if pack.agenda_lines:
            lines.append("【短期任务】")
            lines.extend(pack.agenda_lines)
        if pack.related:
            lines.append("【相关】")
            for index, row in enumerate(pack.related):
                lines.append(_belief_line(row, pack.now, self.tz))
                if index < 2:
                    expansion = _expansion_line(row)
                    if expansion:
                        lines.append(expansion)
        if pack.cross:
            lines.append("【顺带想起来】")
            for row in pack.cross:
                lines.append(_belief_line(row, pack.now, self.tz))
        if pack.due_review:
            lines.append("【到期/待复核】")
            for row in pack.due_review:
                lines.append(_due_review_line(row, pack.now, self.tz))

        body = "\n".join(lines).strip()
        hint = ""
        if pack.hidden_count:
            hint = (
                f"（另有 {pack.hidden_count} 条没展开。问到日期、期限、承诺、进度这类事，"
                f"先调 {tool_hint} 看全再答。）"
            )
        if not body and not hint:
            return ""
        if hint:
            limit = max(0, int(max_chars) - len(hint) - 1)
            body = body[:limit].rstrip()
            return f"{body}\n{hint}".strip()
        return body[:max(0, int(max_chars))]

    # ---------------------------------------------------------------- helpers
    def _embed_for(self, query: str, rows: List[Dict[str, Any]]):
        if self.embed_fn is None:
            return None, None
        try:
            texts = [query] + [str(row.get("label", "")) for row in rows]
            missing = [text for text in texts if text not in self._vec_cache]
            if missing:
                vectors = self.embed_fn(missing)
                for text, vector in zip(missing, vectors):
                    self._vec_cache[text] = vector
            query_vec = self._vec_cache.get(query)
            vectors = {
                str(row.get("id")): self._vec_cache[str(row.get("label", ""))]
                for row in rows
                if str(row.get("label", "")) in self._vec_cache
            }
            return query_vec, vectors
        except Exception:
            logger.debug("embedding failed; falling back to lexical only", exc_info=True)
            return None, None


def _granularity(due_ts: float, tz: str) -> str:
    local = datetime.fromtimestamp(due_ts, _zone(tz))
    return "day" if (local.hour, local.minute, local.second) == (23, 59, 59) else "datetime"


def _due_kind(title: str) -> str:
    if any(word in title for word in ("交", "提交", "截止", "报名", "付款", "还款", "缴费")):
        return "end"
    return "start"


def _lead_seconds(title: str) -> int:
    if any(word in title for word in ("考试", "面试", "答辩", "机票", "签约")):
        return 14 * 86400
    if _due_kind(title) == "end":
        return 7 * 86400
    return DEFAULT_LEAD_S


def _rrule(value: Optional[str]) -> Optional[str]:
    value = (value or "").strip()
    if value.upper().startswith("FREQ="):
        return value
    return None


def _belief_line(row: Dict[str, Any], now: datetime, tz: str) -> str:
    label = str(row.get("label", "")).strip()
    due = row.get("due_at_ts")
    if due:
        human = humanize_due(
            due, now=now, tz=tz,
            kind=str(row.get("time_kind") or "due"),
            raw=str(row.get("time_raw") or ""),
        )
        return f"- {label} — {human}" if human else f"- {label}"
    return f"- {label}"


def _dated_line(row: Dict[str, Any], now: datetime, tz: str) -> str:
    return render_time_line(row, now=now, tz=tz)


def _episode_line(episode: Dict[str, Any]) -> str:
    text = str(episode.get("user_msg") or "").strip().replace("\n", " ")
    if not text:
        return ""
    stamp = str(episode.get("time") or "")[:16].replace("T", " ")
    prefix = f"{stamp} " if stamp else ""
    return f"- {prefix}你：{text[:100]}"


def _expansion_line(row: Dict[str, Any]) -> str:
    reasoning = str(row.get("reasoning") or "").strip()
    evidence = row.get("evidence_texts") or []
    if not reasoning and not evidence:
        return ""
    text = reasoning[:80]
    if evidence:
        text = f"{text}｜原话：{str(evidence[0])[:80]}" if text else f"原话：{str(evidence[0])[:80]}"
    return f"  ↳ {text}"


def _due_review_line(row: Dict[str, Any], now: datetime, tz: str) -> str:
    label = str(row.get("label", "")).strip()
    due = row.get("due_at_ts")
    if due:
        human = humanize_due(due, now=now, tz=tz, kind=str(row.get("time_kind") or "due"),
                             raw=str(row.get("time_raw") or ""))
        return f"- {label} — {human}"
    retention = row.get("retention")
    if retention is not None:
        return f"- {label}（保持率 {float(retention) * 100:.0f}%，需复核）"
    return f"- {label}"
