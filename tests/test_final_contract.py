"""Submission-interface and temporal feature-regression checks."""
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from src.eval_contract import evaluate_files
from src.features import FeatureBuilder


def write(path, headers, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


class FinalContractTests(unittest.TestCase):
    def test_future_weather_and_trips_do_not_change_eta_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'site.json').write_text(json.dumps({'tz': 'Asia/Yekaterinburg',
                                                       'shifts': ['08:00', '20:00']}))
            write(root/'nodes.csv', ['node_id','kind','x_m','y_m','z_m'],
                  [('a','load',0,0,0),('b','junction',50,0,0),('c','dump',100,0,0)])
            write(root/'edges.csv', ['edge_id','from_node','to_node','length_m','surface'],
                  [('e1','a','b',50,'paved'),('e2','b','c',50,'dirt'),
                   ('e3','b','a',50,'paved'),('e4','c','b',50,'dirt')])
            write(root/'vehicles.csv', ['vehicle_id','model','capacity_t'], [('v1','Small',45)])
            write(root/'weather.csv', ['ts','precip_mm_h','visibility_m'],
                  [(0,2.0,1000),(3600,8.0,500)])
            query = dict(query_id='q', trip_id='t1', at_ts=1800, depart_ts=1800,
                         vehicle_id='v1', driver_id='unseen', loaded=0, payload_t=0.0,
                         origin_node='a', dest_node='c', current_edge='e1',
                         done_edges=[], others=[])
            before = FeatureBuilder(root).build(query)
            # A future weather revision and a future trip completion should not enter inference.
            write(root/'weather.csv', ['ts','precip_mm_h','visibility_m'],
                  [(0,2.0,1000),(3600,80000.0,1)])
            (root/'trips.csv').write_text('unknown future trip data', encoding='utf-8')
            after = FeatureBuilder(root).build(query)
            self.assertEqual(set(before), set(after))
            for key in before:
                a, b = before[key], after[key]
                if isinstance(a, float) and math.isnan(a):
                    self.assertTrue(math.isnan(b), key)
                else:
                    self.assertEqual(a, b, key)

    def test_eval_contract_missing_prediction_uses_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root/'nodes.csv',['node_id','kind','x_m','y_m','z_m'],
                  [('a','load',0,0,0),('b','junction',50,0,0),('c','dump',100,0,0)])
            write(root/'edges.csv',['edge_id','from_node','to_node','length_m','surface'],
                  [('e1','a','b',50,'paved'),('e2','b','c',50,'paved')])
            write(root/'vehicles.csv',['vehicle_id','model','capacity_t'], [('v1','Small',45)])
            write(root/'trips.csv',['trip_id','driver_id','loaded','depart_ts','arrive_ts'],
                  [('t1','d1',0,100,130)])
            write(root/'weather.csv',['ts','precip_mm_h','visibility_m'], [(0,0,10000)])
            query = dict(query_id='q1', at_ts=100, trip_id='t1', vehicle_id='v1',
                         driver_id='d1', loaded=0, payload_t=0, origin_node='a',
                         dest_node='c', depart_ts=100, current_edge='e1',
                         done_edges=[], others=[])
            (root/'queries.jsonl').write_text(json.dumps(query)+'\n')
            write(root/'truth.csv',['query_id','remaining_s'],[('q1',30)])
            write(root/'pred.csv',['query_id','p50_s','p90_s'],[])
            report = evaluate_files(root,root/'queries.jsonl',root/'truth.csv',
                                    root/'pred.csv',root/'report.md')
            self.assertEqual(report['missing_or_invalid_predictions'],1)
            self.assertEqual(report['mae_model_s'],report['mae_baseline_s'])
            self.assertIn('### Все запросы', (root/'report.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
