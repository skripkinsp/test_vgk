"""Validate observational ATT against simulator structural potential outcomes.

Runs independent scenarios in a dedicated workspace; NEVER overwrites data/,
truth/ or a trained ETA model. Private outcomes are read ONLY after estimating.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .effect import crossfit_att, load_data
from .simulate import Simulator

SCENARIOS = ((42, 0.0), (43, 0.5), (44, 1.0), (45, 1.5), (46, 2.0))


def evaluate_scenario(data_dir: Path, truth_dir: Path, *, iterations: int,
                      bootstrap: int, seed: int) -> dict:
    # Estimation must happen before any private outcome is opened.
    rec, cols = load_data(data_dir)
    estimate, diagnostics = crossfit_att(rec, cols, iterations=iterations,
                                         bootstrap=bootstrap, seed=seed)
    truth = json.loads((truth_dir / 'effect.json').read_text(encoding='utf-8'))
    with (truth_dir / 'potential_outcomes.csv').open(encoding='utf-8', newline='') as stream:
        potentials = list(csv.DictReader(stream))
    by_rec = {r['rec_id']: r for r in potentials}
    if len(by_rec) != len(potentials) or len(by_rec) != len(rec):
        raise ValueError('Duplicate or missing recommendation potentials')

    accepted_bias = []
    overall_bias = []
    effects_accepted = []
    for r in rec.itertuples(index=False):
        pair = by_rec[str(r.rec_id)]
        treated = int(r.accepted)
        if treated != int(pair['accepted']):
            raise ValueError(f'Treatment mismatch for {r.rec_id}')
        chosen = float(pair['y1_s'] if treated else pair['y0_s'])
        if not math.isclose(chosen, float(r.outcome_s), abs_tol=1e-5):
            raise ValueError(f'Observed potential outcome mismatch for {r.rec_id}')
        effect = float(pair['individual_effect_s'])
        # est_gain_s positive means time saved; effect negative means time saved.
        error = float(r.est_gain_s) + effect
        overall_bias.append(error)
        if treated:
            accepted_bias.append(error)
            effects_accepted.append(effect)
    actual = float(np.mean(effects_accepted))
    if abs(actual - float(truth['att_s'])) > .01:
        raise ValueError('Private ATT does not match potential outcomes')
    low, high = (float(v) for v in estimate['ci95'])
    return {
        'n_recommendations': len(rec),
        'n_accepted': int(truth['n_accepted']),
        'true_att_s': float(truth['att_s']),
        'estimated_att_s': float(estimate['att_s']),
        'att_error_s': round(float(estimate['att_s']) - float(truth['att_s']), 3),
        'ci95_s': [low, high],
        'ci_covers_truth': low <= float(truth['att_s']) <= high,
        'naive_att_s': float(estimate['naive_s']),
        'naive_error_s': round(float(estimate['naive_s']) - float(truth['att_s']), 3),
        'est_gain_bias_accepted_s': round(float(np.mean(accepted_bias)), 3),
        'est_gain_bias_all_s': round(float(np.mean(overall_bias)), 3),
        'diagnostics': diagnostics,
    }


def run_all(work: Path, *, days: int, nodes: int, iterations: int,
            bootstrap: int, reuse: bool, scenarios=SCENARIOS) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    reports = []
    for seed, strength in scenarios:
        directory = work / f'seed_{seed}_bias_{strength:g}'
        data_dir, truth_dir = directory / 'data', directory / 'truth'
        if not reuse or not (truth_dir / 'potential_outcomes.csv').exists():
            print(f'Generating seed={seed}, selection_strength={strength:g}, days={days}', flush=True)
            sim = Simulator(seed=seed, days=days, nodes=nodes, selection_strength=strength)
            truth = sim.run()
            sim.save(data_dir, truth_dir, truth)
        print(f'Estimating ATT: seed={seed}, selection_strength={strength:g}', flush=True)
        result = evaluate_scenario(data_dir, truth_dir, iterations=iterations,
                                   bootstrap=bootstrap, seed=seed)
        result.update({'seed': seed, 'selection_strength': strength})
        reports.append(result)
        print(f"  truth={result['true_att_s']:+.2f}s  "
              f"AIPW={result['estimated_att_s']:+.2f}s  "
              f"covered={result['ci_covers_truth']}", flush=True)
    summary = {
        'scenarios': reports,
        'n_scenarios': len(reports),
        'mae_att_s': round(float(np.mean([abs(r['att_error_s']) for r in reports])), 3),
        'mae_naive_s': round(float(np.mean([abs(r['naive_error_s']) for r in reports])), 3),
        'ci_coverage_fraction': round(float(np.mean([r['ci_covers_truth'] for r in reports])), 3),
        'mean_est_gain_bias_accepted_s': round(float(np.mean([r['est_gain_bias_accepted_s'] for r in reports])), 3),
        'caveats': [
            'Causal outcomes are pre-treatment structural draws, separate from physical fleet loading timestamps; not a full counterfactual fleet replay.',
            'Treatment assignment depends on public pre-decision covariates; ATT identification is conditional on this simplified structural model.',
            'Five scenarios are a diagnostic, not enough for precise 95% CI coverage estimation.',
            'Daily cluster bootstrap keeps nuisance fits fixed; intervals may undercover.',
        ],
    }
    (work / 'effect_validation.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (work / 'effect_validation.csv').open('w', newline='', encoding='utf-8') as stream:
        cols = ['seed','selection_strength','n_recommendations','n_accepted','true_att_s',
                'estimated_att_s','att_error_s','naive_att_s','naive_error_s',
                'ci_covers_truth','est_gain_bias_accepted_s','est_gain_bias_all_s']
        writer = csv.DictWriter(stream, fieldnames=cols, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(reports)
    print('Summary:', json.dumps({k: v for k,v in summary.items() if k != 'scenarios'},
                                ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description='Five independent causal validation scenarios')
    p.add_argument('--work', default='build/effect_validation')
    p.add_argument('--days', type=int, default=21)
    p.add_argument('--nodes', type=int, default=200)
    p.add_argument('--iterations', type=int, default=220)
    p.add_argument('--bootstrap', type=int, default=500)
    p.add_argument('--reuse', action='store_true', help='Reuse existing scenario datasets')
    args = p.parse_args()
    if args.days < 4:
        p.error('--days must be >= 4 for day-fold cross-fitting')
    run_all(Path(args.work), days=args.days, nodes=args.nodes,
            iterations=args.iterations, bootstrap=args.bootstrap, reuse=args.reuse)


if __name__ == '__main__':
    main()
