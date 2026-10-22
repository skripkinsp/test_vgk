"""Assignment baseline: shortest distance from current edge start / fixed speed."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .road_context import RoadContext


def predict_one(query: dict, graph: RoadContext) -> tuple[float, float]:
    edge = graph.edge_by_id[query['current_edge']]
    origin = edge['from_node']   # Requirement: start of the CURRENT edge
    destination = query['dest_node']
    remaining_m = graph.shortest_length(origin, destination)
    speed_kmh = 20.0 if int(query['loaded']) else 30.0
    p50 = remaining_m / (speed_kmh / 3.6)
    return p50, 1.25 * p50


def predict_file(data_dir: str | Path, queries_path: str | Path, out_path: str | Path) -> int:
    graph = RoadContext(data_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with Path(queries_path).open(encoding='utf-8') as source, out_path.open('w', newline='', encoding='utf-8') as dest:
        writer = csv.writer(dest, lineterminator='\n')
        writer.writerow(('query_id', 'p50_s', 'p90_s'))
        for line in source:
            if not line.strip():
                continue
            query = json.loads(line)
            p50, p90 = predict_one(query, graph)
            writer.writerow((query['query_id'], f'{p50:.6f}', f'{p90:.6f}'))
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--queries', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(f'Baseline predictions: {predict_file(args.data, args.queries, args.out)}')


if __name__ == '__main__':
    main()
