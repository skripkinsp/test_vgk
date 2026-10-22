"""Small contract tests for published ETA query and baseline formats."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.baseline import predict_file, predict_one
from src.queries import create_queries
from src.road_context import RoadContext


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(fields)
        writer.writerows(rows)


class EtaIOTests(unittest.TestCase):
    def test_queries_are_temporal_and_baseline_uses_current_edge_start(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_csv(root / 'nodes.csv', ('node_id', 'kind', 'x_m', 'y_m', 'z_m'), [
                ('n0', 'load', 0, 0, 0), ('n1', 'junction', 50, 0, 0),
                ('n2', 'dump', 100, 0, 0)])
            write_csv(root / 'edges.csv', ('edge_id', 'from_node', 'to_node', 'length_m', 'surface'), [
                ('e0', 'n0', 'n1', 50, 'paved'), ('e1', 'n1', 'n2', 50, 'paved'),
                ('e2', 'n0', 'n2', 200, 'gravel'), ('e3', 'n2', 'n1', 50, 'paved'),
                ('e4', 'n1', 'n0', 50, 'paved')])
            write_csv(root / 'vehicles.csv', ('vehicle_id', 'model', 'capacity_t'),
                      [('v1', 'M', 100), ('v2', 'S', 50)])
            write_csv(root / 'trips.csv',
                      ('trip_id', 'vehicle_id', 'driver_id', 'origin_node', 'dest_node',
                       'loaded', 'payload_t', 'depart_ts', 'arrive_ts'), [
                       ('t1', 'v1', 'd1', 'n0', 'n2', 1, 85, 100, 150),
                       ('t2', 'v2', 'd2', 'n2', 'n0', 0, 0, 105, 155)])
            write_csv(root / 'edge_events.csv',
                      ('trip_id', 'seq', 'edge_id', 'enter_ts', 'exit_ts'), [
                       ('t1', 0, 'e0', 100, 110), ('t1', 0, 'e0', 100, 110),
                       ('t1', 1, 'e1', 110, 130), ('t2', 0, 'e3', 105, 117),
                       ('t2', 1, 'e4', 117, 127)])
            report = create_queries(root, 100, 105, root / 'out')
            self.assertEqual(report['trips'], 1)
            self.assertEqual(report['queries'], 3)  # departure + two distinct entries
            queries = [json.loads(s) for s in (root / 'out' / 'queries.jsonl').read_text().splitlines()]
            self.assertEqual([q['at_ts'] for q in queries], [100, 100, 110])
            self.assertEqual(len(queries[-1]['done_edges']), 2)  # retain telemetry duplicate
            self.assertEqual(queries[-1]['others'], [
                {'vehicle_id': 'v2', 'edge_id': 'e3', 'loaded': 0}])
            self.assertTrue(all(q['at_ts'] < 150 for q in queries))
            graph = RoadContext(root)
            self.assertEqual(predict_one(queries[0], graph), (18.0, 22.5))
            self.assertEqual(predict_one(queries[-1], graph), (9.0, 11.25))
            n = predict_file(root, root / 'out' / 'queries.jsonl', root / 'pred.csv')
            self.assertEqual(n, len(queries))
            with (root / 'pred.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 3)


if __name__ == '__main__':
    unittest.main()
