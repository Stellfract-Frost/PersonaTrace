"""
Rhythm engine module.
Extracts temporal patterns from observation logs,
clustering recurring behaviors into rhythms.
"""
import logging
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any
logger = logging.getLogger(__name__)
class RhythmEngine:
    """Temporal rhythm clustering engine.
    Clusters observation_log entries by hour of day.
    Requires at least 3 distinct dates to promote a cluster to a rhythm.
    Supports ±1 hour tolerance and 30-day stale expiry.
    Pure code implementation — no LLM dependency.
    """
    def __init__(self, sim_threshold: float = 0.55):
        """
        Args:
            sim_threshold: clustering similarity threshold
        """
        self.sim_threshold = sim_threshold
        self.weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    def process(
        self,
        data: Dict[str, Any],
        similarity_fn=None,
        hourly_density: Dict = None,
    ) -> None:
        """Process observation log, extract/update rhythms.
        Directly modifies data["persona"]["rhythms"] and data["ephemeral"]["observation_log"].
        Args:
            data: complete profile data dict
            similarity_fn: similarity function (str, str) -> float
            hourly_density: hourly density stats {(weekday, hour): {count, total_len, ...}}
        """
        now = datetime.now()
        ts_now = now.isoformat()
        obs = data.get("ephemeral", {}).get("observation_log", [])
        rhythms = data["persona"].setdefault("rhythms", [])
        # 1. Prune observations older than 60 days
        cutoff_ts = now.timestamp() - 60 * 86400
        kept = []
        for o in obs:
            try:
                if datetime.fromisoformat(o["ts"]).timestamp() >= cutoff_ts:
                    kept.append(o)
            except (ValueError, KeyError):
                kept.append(o)
        data["ephemeral"]["observation_log"] = kept
        # 2. Group by hour
        by_hour: Dict[int, List] = defaultdict(list)
        for i, o in enumerate(kept):
            h = o.get("hour", -1)
            if h >= 0:
                by_hour[h].append((i, o))
        # 3. Cluster within ±1 hour window; require ≥3 distinct dates
        promoted_indices = set()
        for hour in range(24):
            candidates = []
            for h in (hour - 1, hour, hour + 1):
                candidates.extend(by_hour.get(h % 24, []))
            if len(candidates) < 3:
                continue
            used = [False] * len(candidates)
            for i in range(len(candidates)):
                if used[i]:
                    continue
                cluster = [candidates[i]]
                used[i] = True
                for j in range(i + 1, len(candidates)):
                    if used[j]:
                        continue
                    if similarity_fn:
                        score = similarity_fn(
                            candidates[i][1].get("label", ""),
                            candidates[j][1].get("label", ""),
                        )
                    else:
                        # fallback to simple string matching
                        a = candidates[i][1].get("label", "").lower()
                        b = candidates[j][1].get("label", "").lower()
                        score = 1.0 if a == b else (0.8 if a in b or b in a else 0.0)
                    if score >= self.sim_threshold:
                        cluster.append(candidates[j])
                        used[j] = True
                days = {c[1].get("ts", "")[:10] for c in cluster if c[1].get("ts", "")}
                if len(cluster) < 3 or len(days) < 3:
                    continue
                self._promote_cluster(
                    cluster, rhythms, hourly_density, now, ts_now
                )
                for idx, _ in cluster:
                    promoted_indices.add(idx)
        # 4. Remove promoted observations
        data["ephemeral"]["observation_log"] = [
            o for i, o in enumerate(kept) if i not in promoted_indices
        ]
        # 5. Clean up rhythms not confirmed in 30 days
        data["persona"]["rhythms"] = [
            r for r in rhythms if not self._is_stale(r, now)
        ]
    def _promote_cluster(
        self, cluster, rhythms, hourly_density, now, ts_now
    ):
        """Promote a cluster to a rhythm, or update an existing one."""
        best = max(cluster, key=lambda x: x[1].get("confidence", 0))
        label = best[1]["label"]
        occurrences = len(cluster)
        new_conf = min(0.92, 0.50 + occurrences * 0.10)
        last_ts = max(c[1].get("ts", "") for c in cluster)
        # determine dominant hour and weekday
        hour_counts = defaultdict(int)
        for _, item in cluster:
            hour_counts[item.get("hour", -1)] += 1
        dominant_hour = max(hour_counts, key=hour_counts.get)
        try:
            latest_dt = datetime.fromisoformat(last_ts)
            dominant_wd = latest_dt.weekday()
        except (ValueError, TypeError):
            dominant_wd = now.weekday()
        wd_name = self.weekdays[dominant_wd]
        # look up density stats
        d = (hourly_density or {}).get((dominant_wd, dominant_hour), {})
        d_count = d.get("count", 0)
        d_total = d.get("total_len", 0)
        d_ask = d.get("ask_count", 0)
        d_avg = round(d_total / d_count) if d_count > 0 else 0
        d_askr = round(d_ask / d_count, 2) if d_count > 0 else 0
        # find existing rhythm and update
        found = False
        for r in rhythms:
            if label.lower() in r.get("label", "").lower() or \
                r.get("label", "").lower() in label.lower():
                r["confidence"] = min(0.95, max(r.get("confidence", 0), new_conf))
                r["occurrences"] = r.get("occurrences", 0) + occurrences
                r["last_seen"] = max(r.get("last_seen", ""), last_ts)
                r["weekday"] = dominant_wd
                r["hour"] = dominant_hour
                r["updated"] = ts_now
                r["msg_count"] = d_count
                r["avg_len"] = d_avg
                r["ask_ratio"] = d_askr
                found = True
                break
        if not found:
            rhythms.append({
                "label": label,
                "weekday": dominant_wd,
                "hour": dominant_hour,
                "confidence": round(new_conf, 2),
                "occurrences": occurrences,
                "last_seen": last_ts,
                "time_note": f"{wd_name} {dominant_hour:02d}:00",
                "updated": ts_now,
                "msg_count": d_count,
                "avg_len": d_avg,
                "ask_ratio": d_askr,
            })
    @staticmethod
    def _is_stale(rhythm: Dict, now: datetime) -> bool:
        """Check if a rhythm hasn't been confirmed in 30 days."""
        try:
            last = datetime.fromisoformat(rhythm.get("last_seen", ""))
            return (now - last).days > 30
        except (ValueError, TypeError):
            return False
