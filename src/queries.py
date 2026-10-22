"""Build ETA queries using only published trips/telemetry.

Data fields are historical, but each query contains only observations available
at or before its own at_ts. Missing telemetry cannot be reconstructed perfectly.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .road_context import RoadContext


def parse_time(value: str | None) -> int | None:
    if not value:
        return None
    if value.lstrip('-').isdigit():
        return int(value)
    value = value.replace('Z', '+00:00')
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def load_trips(path: Path) -> dict[str, dict]:
    with path.open(newline='', encoding='utf-8') as f:
        return {row['trip_id']: row for row in csv.DictReader(f)}


def output_paths(out: str | Path) -> tuple[Path, Path]:
    path = Path(out)
    if path.suffix == '.jsonl':
        return path, path.with_name('truth.csv')
    return path / 'queries.jsonl', path / 'truth.csv'


def create_queries(data_dir: str | Path, start_ts: int | None, end_ts: int | None,
                   out: str | Path) -> dict[str, int]:
    root = Path(data_dir)
    graph = RoadContext(root)
    trips = load_trips(root / 'trips.csv')
    selected = {
        trip_id: trip for trip_id, trip in trips.items()
        if (start_ts is None or int(trip['depart_ts']) >= start_ts)
        and (end_ts is None or int(trip['depart_ts']) < end_ts)
    }
    observed = defaultdict(list)
    # All trips (not just selected) are needed to determine current positions of others.
    occupancy_events = []
    with (root / 'edge_events.csv').open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            trip_id = row['trip_id']
            if trip_id not in trips:
                continue
            item = (int(row['seq']), int(row['enter_ts']), int(row['exit_ts']), row['edge_id'])
            if item[3] not in graph.edge_by_id or item[2] <= item[1]:
                continue
            if trip_id in selected:
                observed[trip_id].append(item)
            occupancy_events.append((item[1], item[2], trips[trip_id]['vehicle_id'], item[3], int(trips[trip_id]['loaded'])))
    occupancy_events.sort()

    # Query specification: (at_ts, trip_id, kind, seq, current_edge)
    # kind 0 = departure; kind 1 = edge entrance.
    events = []
    fallback_departures = 0
    for trip_id, trip in selected.items():
        depart, arrive = int(trip['depart_ts']), int(trip['arrive_ts'])
        if arrive <= depart:
            continue
        rows = observed[trip_id]
        first = next((r[3] for r in rows if r[0] == 0), None)
        if first is None:
            first = graph.first_edge_guess(trip['origin_node'], trip['dest_node'])
            fallback_departures += 1
        events.append((depart, trip_id, 0, -1, first))
        # Duplicate telemetry events must NOT produce duplicate forecast triggers.
        unique = set(rows)
        for seq, enter, exit_, edge_id in unique:
            if depart <= enter < arrive:
                events.append((enter, trip_id, 1, seq, edge_id))
        rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    events.sort()

    q_path, truth_path = output_paths(out)
    q_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.parent.mkdir(parents=True, exist_ok=True)

    # Temporal sweep to recover positions from OBSERVED telemetry only.
    # Duplicate rows and shifted clocks may produce conflicts; latest entered
    # observed segment wins for a vehicle.
    active = {}                  # vehicle_id -> (serial, edge_id, loaded)
    leaving = []                 # (exit_ts, serial, vehicle_id)
    enter_index = 0
    n = 0
    with q_path.open('w', encoding='utf-8') as query_file, \
         truth_path.open('w', encoding='utf-8', newline='') as target_file:
        target = csv.writer(target_file, lineterminator='\n')
        target.writerow(('query_id', 'remaining_s'))
        for at, trip_id, kind, seq, current_edge in events:
            while enter_index < len(occupancy_events) and occupancy_events[enter_index][0] <= at:
                start, stop, vehicle_id, edge_id, loaded = occupancy_events[enter_index]
                serial = enter_index
                enter_index += 1
                if stop > at:
                    active[vehicle_id] = (serial, edge_id, loaded)
                    heapq.heappush(leaving, (stop, serial, vehicle_id))
            while leaving and leaving[0][0] <= at:
                _, serial, vehicle_id = heapq.heappop(leaving)
                if vehicle_id in active and active[vehicle_id][0] == serial:
                    del active[vehicle_id]

            trip = selected[trip_id]
            vehicle_id = trip['vehicle_id']
            done = [dict(edge_id=edge_id, enter_ts=enter, exit_ts=exit_)
                    for observed_seq, enter, exit_, edge_id in observed[trip_id]
                    if observed_seq < seq and exit_ <= at]
            others = [dict(vehicle_id=v, edge_id=val[1], loaded=int(val[2]))
                      for v, val in sorted(active.items()) if v != vehicle_id]
            query_id = f'q-{n:08d}'
            query = dict(query_id=query_id, at_ts=at, trip_id=trip_id,
                         vehicle_id=vehicle_id, driver_id=trip['driver_id'],
                         loaded=int(trip['loaded']), payload_t=float(trip['payload_t']),
                         origin_node=trip['origin_node'], dest_node=trip['dest_node'],
                         depart_ts=int(trip['depart_ts']), current_edge=current_edge,
                         done_edges=done, others=others)
            query_file.write(json.dumps(query, ensure_ascii=False, separators=(',', ':')) + '\n')
            target.writerow((query_id, int(trip['arrive_ts']) - at))
            n += 1
    return {'trips': len(selected), 'queries': n, 'inferred_first_edges': fallback_departures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--from', dest='start', default='')
    parser.add_argument('--to', dest='end', default='')
    parser.add_argument('--out', default='build/queries')
    args = parser.parse_args()
    start, end = parse_time(args.start), parse_time(args.end)
    if start is not None and end is not None and start >= end:
        parser.error('FROM must precede TO')
    print(json.dumps(create_queries(args.data, start, end, args.out), indent=2))


if __name__ == '__main__':
    main()
