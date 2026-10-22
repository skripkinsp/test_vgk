import unittest

import numpy as np
import pandas as pd

from src.effect import crossfit_att


class EffectTests(unittest.TestCase):
    @staticmethod
    def synthetic_zero_effect(seed=12):
        rng = np.random.default_rng(seed)
        days = np.repeat(np.arange(12), 60)
        n = len(days)
        x = rng.normal(size=n)
        # Randomized policy, no treatment effect; outcome depends on measured X.
        treatment = rng.binomial(1, .45, size=n)
        outcome = 300 + 70*x + rng.normal(0, 25, size=n)
        df = pd.DataFrame({
            'accepted': treatment, 'outcome_s': outcome, 'day_index': days,
            'vehicle_id': np.where(x > 0, 'v-01', 'v-02'),
            'model': np.where(x > 0, 'Heavy', 'Small'),
            'planned_node': 'load-a', 'recommended_node': 'load-b',
            'capacity_t': 45.0, 'est_gain_s': 10*x,
            'hour_sin': np.sin(x), 'hour_cos': np.cos(x),
            'night': 0, 'minutes_to_shift': 100.0,
            'precip_mm_h': 0.0, 'visibility_m': 10000.0,
            'planned_arrivals_1h': 2.0, 'suggested_arrivals_1h': 2.0,
        })
        cols = ['vehicle_id','model','planned_node','recommended_node','capacity_t',
                'est_gain_s','hour_sin','hour_cos','night','minutes_to_shift',
                'day_index','precip_mm_h','visibility_m','planned_arrivals_1h',
                'suggested_arrivals_1h']
        return df, cols

    def test_zero_effect_ci_contains_zero(self):
        data, features = self.synthetic_zero_effect()
        result, diagnostics = crossfit_att(data, features, iterations=30, seed=42, bootstrap=120)
        self.assertLess(result['ci95'][0], 0)
        self.assertGreater(result['ci95'][1], 0)
        self.assertEqual(diagnostics['n'], 720)

    def test_reproducible(self):
        data, features = self.synthetic_zero_effect()
        first, _ = crossfit_att(data, features, iterations=25, seed=42, bootstrap=100)
        second, _ = crossfit_att(data, features, iterations=25, seed=42, bootstrap=100)
        self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
