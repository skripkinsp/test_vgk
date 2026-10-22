"""ETA infrastructure tests that do not require the 21-day simulator data."""
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from src.baseline import predict_one as baseline_one
from src.eta_runtime import EtaPredictor
from src.road_context import RoadContext


def write_csv(path, columns, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(rows)


class TestEtaInfrastructure(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        write_csv(self.root/'nodes.csv', ['node_id','kind','x_m','y_m','z_m'],
                  [('a','load',0,0,0), ('b','junction',50,0,0), ('c','dump',100,0,0)])
        write_csv(self.root/'edges.csv', ['edge_id','from_node','to_node','length_m','surface'],
                  [('e1','a','b',50,'paved'),('e2','b','c',50,'dirt')])
        write_csv(self.root/'vehicles.csv',['vehicle_id','model','capacity_t'], [('v1','Small',45)])
        self.query = dict(query_id='q1',at_ts=120,trip_id='t1',vehicle_id='v1',driver_id='d1',
                          loaded=0,payload_t=0,origin_node='a',dest_node='c',depart_ts=120,
                          current_edge='e1',done_edges=[],others=[])

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_model_falls_back(self):
        predictor = EtaPredictor(self.root, self.root/'missing')
        a,b,used = predictor.predict_one(self.query)
        self.assertTrue(used)
        self.assertEqual((a,b),baseline_one(self.query,RoadContext(self.root)))

    def test_unknown_driver_vehicle_with_missing_model(self):
        predictor = EtaPredictor(self.root,self.root/'missing')
        q = dict(self.query,driver_id='unseen-driver',vehicle_id='unseen-vehicle')
        p50,p90,used = predictor.predict_one(q)
        self.assertTrue(used)
        self.assertGreaterEqual(p90,p50)

    def test_corrupted_ml_prediction_falls_back(self):
        predictor = EtaPredictor(self.root,self.root/'missing')
        class BadBuilder:
            def build(self, q):
                raise ValueError('broken feature')
        predictor.models = [object(),object()]
        predictor.builder = BadBuilder()
        self.assertEqual(predictor.predict_one(self.query),(*baseline_one(self.query,predictor.graph),True))

    def test_unknown_graph_edge_is_explicit_error(self):
        predictor = EtaPredictor(self.root,self.root/'missing')
        with self.assertRaises(KeyError):
            predictor.predict_one(dict(self.query,current_edge='bad'))

    def test_future_files_not_used_by_fallback(self):
        predictor = EtaPredictor(self.root,self.root/'missing')
        before = predictor.predict_one(self.query)
        (self.root/'trips.csv').write_text('malicious future contents',encoding='utf-8')
        (self.root/'edge_events.csv').write_text('malicious future contents',encoding='utf-8')
        self.assertEqual(before,predictor.predict_one(self.query))


if __name__=='__main__':
    unittest.main()
