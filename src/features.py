"""ETA features from query + allowed public data only.

No trips.csv, edge_events.csv, future weather or simulator state is read here.
"""
from __future__ import annotations

import bisect
import csv
import json
import math
import statistics
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from .graph import k_shortest_paths
from .road_context import RoadContext

K_ROUTES = 5
CAT_FEATURES = ["vehicle_model", "driver_id"]
SURFACES = ("paved", "gravel", "dirt")


class FeatureBuilder:
    def __init__(self, data_dir: str | Path):
        self.road = RoadContext(data_dir)
        root = Path(data_dir)
        with (root / "site.json").open(encoding="utf-8") as f:
            site = json.load(f)
        self.tz = ZoneInfo(site["tz"])
        self.shifts = [60 * int(s[:2]) + int(s[3:]) for s in site["shifts"]]
        with (root / "weather.csv").open(newline="", encoding="utf-8") as f:
            weather = sorted(csv.DictReader(f), key=lambda r: int(r["ts"]))
        self.weather_ts = [int(r["ts"]) for r in weather]
        self.precip = [float(r["precip_mm_h"]) for r in weather]
        self.visibility = [float(r["visibility_m"]) for r in weather]
        self.rain_prefix = [0.0]
        for x in self.precip:
            self.rain_prefix.append(self.rain_prefix[-1] + x)

        # The imported K-shortest-path algorithm uses integer node/edge indexes.
        self.node_idx = {n["node_id"]: i for i, n in enumerate(self.road.nodes)}
        self.edges = []
        self.adj = {i: [] for i in range(len(self.road.nodes))}
        self.edge_index = {}
        heights = {n["node_id"]: float(n["z_m"]) for n in self.road.nodes}
        for raw in self.road.edges:
            u, v = self.node_idx[raw["from_node"]], self.node_idx[raw["to_node"]]
            length = raw["length_m"]
            edge = dict(u=u, v=v, length=length, surface=raw["surface"],
                        slope=(heights[raw["to_node"]] - heights[raw["from_node"]]) / length,
                        edge_id=raw["edge_id"])
            idx = len(self.edges)
            self.edges.append(edge)
            self.edge_index[raw["edge_id"]] = idx
            self.adj[u].append((v, idx))
        for a in self.adj.values():
            a.sort()

    def _edge_seconds(self, edge: dict, loaded: int) -> float:
        # Approximate, public-data-only engineering prior; NOT simulator coefficients.
        surface_factor = {"paved": 1.0, "gravel": .84, "dirt": .70}.get(edge["surface"], .75)
        uphill = max(edge["slope"], 0.0)
        downhill = max(-edge["slope"], 0.0)
        slope_factor = max(.40, 1.0 - (6.5 if loaded else 3.2) * uphill - 1.0 * downhill)
        speed_kmh = (20.0 if loaded else 30.0) * surface_factor * slope_factor
        return 3.6 * edge["length"] / max(4.0, speed_kmh)

    @lru_cache(maxsize=12000)
    def _suffix_routes(self, start: int, dest: int, loaded: int) -> tuple[tuple[int, ...], ...]:
        weight = lambda e: self._edge_seconds(e, loaded)
        return tuple(k_shortest_paths(self.adj, self.edges, start, dest, weight, k=K_ROUTES))

    def _rain_features(self, at: int) -> tuple[float, float, float, float]:
        i = bisect.bisect_right(self.weather_ts, at) - 1
        if i < 0:
            return (0.0, 0.0, 0.0, 10000.0)
        # Hourly measurements: previous 3/6 observations, never a future row.
        rain3 = self.rain_prefix[i + 1] - self.rain_prefix[max(0, i - 2)]
        rain6 = self.rain_prefix[i + 1] - self.rain_prefix[max(0, i - 5)]
        return self.precip[i], rain3, rain6, self.visibility[i]

    def build(self, q: dict) -> dict:
        at, loaded = int(q["at_ts"]), int(q["loaded"])
        edge_id = q["current_edge"]
        current_idx = self.edge_index[edge_id]
        current = self.edges[current_idx]
        destination = self.node_idx[q["dest_node"]]
        suffixes = self._suffix_routes(current["v"], destination, loaded)
        routes = [(current_idx,) + s for s in suffixes]
        if not routes:
            raise ValueError(f"No route from {edge_id} to {q['dest_node']}")

        vehicle = self.road.vehicles.get(q["vehicle_id"], {})
        capacity = float(vehicle.get("capacity_t", 0) or 0)
        moment = datetime.fromtimestamp(at, self.tz)
        minutes = moment.hour * 60 + moment.minute
        to_shift = min((s - minutes) % 1440 for s in self.shifts)
        from_shift = min((minutes - s) % 1440 for s in self.shifts)
        rain, rain3, rain6, vis = self._rain_features(at)

        # Occupancy is observable at at_ts; no other vehicle's future route.
        occupancy: dict[tuple[int, int], int] = {}
        for other in q.get("others", []):
            other_edge = self.road.edge_by_id.get(other.get("edge_id"))
            if other_edge is None:
                continue
            u, v = self.node_idx[other_edge["from_node"]], self.node_idx[other_edge["to_node"]]
            physical = (min(u, v), max(u, v))
            occupancy[physical] = occupancy.get(physical, 0) + 1

        seen = set()
        observed = []
        for done in q.get("done_edges", []):
            key = (done["edge_id"], int(done["enter_ts"]), int(done["exit_ts"]))
            if key in seen or key[0] not in self.edge_index:
                continue
            seen.add(key)
            duration = key[2] - key[1]
            if 0 < duration <= 7200 and key[2] <= at:
                edge = self.edges[self.edge_index[key[0]]]
                observed.append((edge["length"], duration))
        done_dist = sum(x for x, _ in observed)
        done_duration = sum(t for _, t in observed)
        done_speeds = [3.6 * dist / seconds for dist, seconds in observed]

        x = dict(
            vehicle_model=str(vehicle.get("model", "UNKNOWN")),
            driver_id=str(q.get("driver_id") or "UNKNOWN"),
            capacity_t=capacity,
            loaded=loaded,
            payload_t=float(q.get("payload_t", 0)),
            payload_fraction=float(q.get("payload_t", 0)) / capacity if capacity > 0 else 0.0,
            hour=moment.hour, weekday=moment.weekday(),
            night=int(moment.hour < 6 or moment.hour >= 22),
            minutes_to_shift=to_shift, minutes_since_shift=from_shift,
            precip_mm_h=rain, rain_3h_mm=rain3, rain_6h_mm=rain6,
            visibility_m=vis,
            current_surface={"paved": 0, "gravel": 1, "dirt": 2}.get(current["surface"], 3),
            current_slope=current["slope"],
            current_edge_traffic=occupancy.get((min(current["u"], current["v"]),
                                                max(current["u"], current["v"])), 0),
            other_vehicles=len(q.get("others", [])),
            elapsed_s=max(0, at - int(q["depart_ts"])),
            done_count=len(observed), done_distance_m=done_dist,
            observed_speed_kmh=3.6 * done_dist / done_duration if done_duration else math.nan,
            median_edge_speed_kmh=statistics.median(done_speeds) if done_speeds else math.nan,
            shortest_distance_m=self.road.shortest_length(self.road.edges[current_idx]["from_node"],
                                                           q["dest_node"]),
            route_count=len(routes),
        )
        x["baseline_eta_s"] = x["shortest_distance_m"] / ((20 if loaded else 30) / 3.6)

        for i in range(K_ROUTES):
            prefix = f"r{i+1}_"
            if i >= len(routes):
                for name in ("eta_s", "length_m", "dirt_m", "gravel_m", "uphill_m",
                             "downhill_m", "max_uphill", "traffic", "traffic_edges"):
                    x[prefix + name] = math.nan
                continue
            route = routes[i]
            edges = [self.edges[j] for j in route]
            x[prefix + "eta_s"] = sum(self._edge_seconds(e, loaded) for e in edges)
            x[prefix + "length_m"] = sum(e["length"] for e in edges)
            x[prefix + "dirt_m"] = sum(e["length"] for e in edges if e["surface"] == "dirt")
            x[prefix + "gravel_m"] = sum(e["length"] for e in edges if e["surface"] == "gravel")
            x[prefix + "uphill_m"] = sum(e["length"] * max(e["slope"], 0) for e in edges)
            x[prefix + "downhill_m"] = sum(e["length"] * max(-e["slope"], 0) for e in edges)
            x[prefix + "max_uphill"] = max(max(e["slope"], 0) for e in edges)
            counts = [occupancy.get((min(e["u"], e["v"]), max(e["u"], e["v"])), 0) for e in edges]
            x[prefix + "traffic"] = sum(counts)
            x[prefix + "traffic_edges"] = sum(n > 0 for n in counts)
        return x
