"""
event_memory.py — reused from kitchen system, traffic-adapted fields.
"""
import json
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass, asdict


@dataclass
class TrafficEvent:
    id: str
    timestamp: datetime
    location: str
    event_type: str          # "congestion" | "violation" | "normal" | "high_risk"
    confidence: float
    vehicle_count: int
    light_state: str
    objects: List[str]
    speed_kmh: float = 0.0
    frame_path: Optional[str] = None


class TrafficEventMemory:
    def __init__(self, storage_path="traffic_events.json"):
        self.path = storage_path
        self.events = self._load()

    def _load(self):
        try:
            with open(self.path) as f:
                data = json.load(f)
                return [
                    TrafficEvent(
                        **{**d, "timestamp": datetime.fromisoformat(d["timestamp"])}
                    )
                    for d in data
                ]
        except Exception:
            return []

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(
                [{**asdict(e), "timestamp": e.timestamp.isoformat()} for e in self.events],
                f, indent=2,
            )

    def add_event(self, event: TrafficEvent):
        self.events.append(event)
        self._save()

    def query(self, location=None, start_time=None, end_time=None, limit=10):
        results = self.events
        if location:
            results = [e for e in results if e.location == location]
        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]
        return sorted(results, key=lambda x: x.timestamp, reverse=True)[:limit]

    def generate_context(self, hours=1, limit=5):
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = self.query(start_time=cutoff, limit=limit)
        if not recent:
            return "No recent traffic events."
        lines = ["Recent traffic events:"]
        for e in recent:
            lines.append(
                f"{e.timestamp.strftime('%H:%M')} — {e.event_type} at {e.location}, "
                f"vehicles: {e.vehicle_count}, light: {e.light_state}, "
                f"speed: {e.speed_kmh:.0f} km/h"
            )
        return "\n".join(lines)

    def congestion_summary(self, hours=1):
        """Returns average vehicle count over the past N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = self.query(start_time=cutoff, limit=100)
        if not recent:
            return 0.0
        return sum(e.vehicle_count for e in recent) / len(recent)
