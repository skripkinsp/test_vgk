"""Paired causal outcomes are consistent with the realized event simulation."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.simulate import Simulator


class TestCausalTruth(unittest.TestCase):
    def test_generator_zero_causal_effect(self):
        sim = Simulator(seed=123, days=1, nodes=80, causal_effect_scale=0.0)
        truth = sim.run()
        self.assertEqual(truth['att_s'], 0.0)
        self.assertGreater(truth['n_accepted'], 0)
        self.assertTrue(all(abs(rec['_y1']-rec['_y0']) < 1e-9
                            for rec in sim.rec_rows))
        sim.edge_stream.close()

    def test_consistency_and_att(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sim = Simulator(seed=123, days=1, nodes=80)
            truth = sim.run()
            sim.save(root / 'data', root / 'truth', truth)
            with (root / 'data' / 'recommendations.csv').open(newline='') as f:
                recs = list(csv.DictReader(f))
            with (root / 'truth' / 'potential_outcomes.csv').open(newline='') as f:
                pairs = list(csv.DictReader(f))
            self.assertEqual(len(recs), len(pairs))
            accepted_effects = []
            for rec, pair in zip(recs, pairs):
                self.assertEqual(rec['rec_id'], pair['rec_id'])
                selected = 'y1_s' if int(rec['accepted']) else 'y0_s'
                self.assertAlmostEqual(float(rec['outcome_s']), float(pair[selected]), places=5)
                if int(rec['accepted']):
                    accepted_effects.append(float(pair['individual_effect_s']))
            self.assertTrue(accepted_effects)
            self.assertAlmostEqual(sum(accepted_effects) / len(accepted_effects), truth['att_s'], places=3)
            self.assertEqual(json.loads((root / 'truth' / 'effect.json').read_text()), truth)


if __name__ == '__main__':
    unittest.main()
