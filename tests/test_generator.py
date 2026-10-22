"""Generator-stage checks. ETA leakage and causal-coverage tests come later."""
import csv
import hashlib
import random
import tempfile
import unittest
from pathlib import Path

from src.graph import generate_graph, k_shortest_paths, shortest_path
from src.simulate import Simulator
from src.validate import check


class GraphTests(unittest.TestCase):
    def test_graph_is_connected_and_has_valid_slopes(self):
        nodes, edges, adj, loads, dumps = generate_graph(random.Random(42),200)
        self.assertEqual(len(nodes),202)
        self.assertEqual(len(loads),3)
        self.assertEqual(len(dumps),2)
        self.assertEqual({e['surface'] for e in edges},{'paved','gravel','dirt'})
        self.assertTrue(all(e['length_m']==50 for e in edges))
        self.assertTrue(all(abs(e['slope']) <= .080001 for e in edges))
        for start in (0,loads[0],dumps[1]):
            todo=[start]; seen={start}
            for node in todo:
                for v,_ in adj[node]:
                    if v not in seen:
                        seen.add(v); todo.append(v)
            self.assertEqual(len(seen),len(nodes))
        # The dump nodes are dead ends in the undirected sense.
        self.assertTrue(all(len(adj[d])==1 for d in dumps))

    def test_yen_returns_sorted_distinct_simple_paths(self):
        _, edges, adj, loads, dumps = generate_graph(random.Random(42),200)
        weight=lambda e:e['length_m']*(1.0 if e['surface']=='paved' else 1.2)
        routes=k_shortest_paths(adj,edges,loads[0],dumps[0],weight,5)
        self.assertTrue(1<=len(routes)<=5)
        self.assertEqual(len(set(routes)),len(routes))
        costs=[sum(weight(edges[idx]) for idx in route) for route in routes]
        self.assertEqual(costs,sorted(costs))
        for route in routes:
            seen={loads[0]}; node=loads[0]
            for idx in route:
                e=edges[idx]
                self.assertEqual(node,e['u'])
                node=e['v']
                self.assertNotIn(node,seen)
                seen.add(node)
            self.assertEqual(node,dumps[0])


class SimulationTests(unittest.TestCase):
    @staticmethod
    def generate(folder,seed=19,days=2):
        sim=Simulator(seed=seed,days=days,nodes=120)
        truth=sim.run()
        data=Path(folder)/'data'; private=Path(folder)/'truth'
        sim.save(data,private,truth)
        check(data,private,days)
        return data,private,sim

    def test_format_routes_and_recommendations(self):
        with tempfile.TemporaryDirectory() as temp:
            data,private,sim=self.generate(temp)
            self.assertGreater(sim.breakdowns+sim.stuck_events+sim.congested_entries,0)
            with (data/'trips.csv').open() as f:
                trips={r['trip_id']:r for r in csv.DictReader(f)}
            with (data/'recommendations.csv').open() as f:
                recs=list(csv.DictReader(f))
            self.assertGreater(len(recs),0)
            for r in recs:
                trip=trips[r['trip_id']]
                self.assertEqual(trip['loaded'],'0')
                self.assertIn(trip['origin_node'],{sim.nodes[d]['node_id'] for d in sim.dumps})
                self.assertEqual(trip['dest_node'],r['recommended_node'] if r['accepted']=='1' else r['planned_node'])
            self.assertGreater(sum(1 for v in trips.values() if v['loaded']=='1'),0)

    def test_seed_reproducibility_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            da,ta,_=self.generate(a,seed=73,days=1)
            db,tb,_=self.generate(b,seed=73,days=1)
            for file in list(da.iterdir())+list(ta.iterdir()):
                counterpart=(db if file.parent==da else tb)/file.name
                self.assertEqual(hashlib.sha256(file.read_bytes()).digest(),
                                 hashlib.sha256(counterpart.read_bytes()).digest(),file.name)

    def test_speed_depends_on_load_and_grade(self):
        sim=Simulator(seed=5,days=1,nodes=100)
        uphill={'length_m':50,'surface':'gravel','slope':.07}
        flat={'length_m':50,'surface':'gravel','slope':0}
        self.assertLess(sim.norm_speed(uphill,1,True),sim.norm_speed(flat,1,True))
        self.assertLess(sim.norm_speed(uphill,1,True),sim.norm_speed(uphill,1,False))
        sim.edge_stream.close()


if __name__=='__main__':
    unittest.main()
