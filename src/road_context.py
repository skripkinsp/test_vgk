"""Read-only road graph reconstructed from the published CSV files.

No simulator objects or simulator-only attributes are used here.
"""
from __future__ import annotations

import csv
import heapq
from pathlib import Path


class RoadContext:
    def __init__(self, data_dir: str | Path):
        root = Path(data_dir)
        with (root / 'nodes.csv').open(newline='', encoding='utf-8') as f:
            self.nodes = list(csv.DictReader(f))
        with (root / 'edges.csv').open(newline='', encoding='utf-8') as f:
            self.edges = list(csv.DictReader(f))
        with (root / 'vehicles.csv').open(newline='', encoding='utf-8') as f:
            self.vehicles = {r['vehicle_id']: r for r in csv.DictReader(f)}

        self.node_ids = {r['node_id'] for r in self.nodes}
        self.edge_by_id = {}
        self.outgoing = {node: [] for node in self.node_ids}
        self.incoming = {node: [] for node in self.node_ids}
        for edge in self.edges:
            src, dst = edge['from_node'], edge['to_node']
            if src not in self.node_ids or dst not in self.node_ids:
                raise ValueError(f"Unknown node in edge {edge['edge_id']}")
            edge['length_m'] = float(edge['length_m'])
            if edge['length_m'] <= 0:
                raise ValueError(f"Non-positive length: {edge['edge_id']}")
            if edge['edge_id'] in self.edge_by_id:
                raise ValueError(f"Duplicate edge: {edge['edge_id']}")
            self.edge_by_id[edge['edge_id']] = edge
            self.outgoing[src].append(edge)
            self.incoming[dst].append(edge)
        for value in self.outgoing.values():
            value.sort(key=lambda edge: edge['edge_id'])
        self._distance_cache = {}

    def distances_to(self, destination: str) -> dict[str, float]:
        """One reverse Dijkstra per destination, cached for later requests."""
        if destination not in self.node_ids:
            raise ValueError(f'Unknown destination: {destination}')
        if destination not in self._distance_cache:
            dist = {destination: 0.0}
            queue = [(0.0, destination)]
            while queue:
                length, v = heapq.heappop(queue)
                if length > dist[v]:
                    continue
                for edge in self.incoming[v]:
                    u = edge['from_node']
                    candidate = length + edge['length_m']
                    if candidate < dist.get(u, float('inf')):
                        dist[u] = candidate
                        heapq.heappush(queue, (candidate, u))
            self._distance_cache[destination] = dist
        return self._distance_cache[destination]

    def shortest_length(self, origin: str, destination: str) -> float:
        distance = self.distances_to(destination).get(origin)
        if distance is None:
            raise ValueError(f'No route from {origin} to {destination}')
        return distance

    def first_edge_guess(self, origin: str, destination: str) -> str:
        """Fallback when the first observed telemetry event is missing."""
        dist = self.distances_to(destination)
        candidates = [e for e in self.outgoing[origin] if e['to_node'] in dist]
        if not candidates:
            raise ValueError(f'No outgoing route from {origin} to {destination}')
        edge = min(candidates, key=lambda e: (e['length_m'] + dist[e['to_node']], e['edge_id']))
        return edge['edge_id']
