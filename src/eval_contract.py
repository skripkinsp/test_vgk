"""Submission interface: QUERIES + TRUTH + PRED -> eval/report.md.

The evaluator may use completed trips/telemetry for diagnostic *slices* only.
Its baseline is recomputed from the public graph. No truth/private causal data
are used to produce predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import predict_one as baseline_predict_one
from .eval import _unique_csv, load_context, load_queries, metrics
from .road_context import RoadContext


def _baseline_frame(data_dir: Path, queries: Path) -> pd.DataFrame:
    graph = RoadContext(data_dir)
    rows = []
    with queries.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                p50, p90 = baseline_predict_one(q, graph)
                rows.append((q['query_id'], p50, p90))
    return pd.DataFrame(rows, columns=['query_id', 'p50_s_base', 'p90_s_base'])


def _table(df: pd.DataFrame, label: str) -> list[str]:
    lines = [f'### {label}', '', '| Модель | N | MAE p50 (с) | Pinball 0.5 (с) | Pinball 0.9 (с) | Coverage p90 | p90 < p50 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for short, name in [('base', 'Дейкстра'), ('cb', 'CatBoost / fallback')]:
        m = metrics(df, short)
        lines.append(f'| {name} | {m["n"]} | {m["mae_s"]:.3f} | {m["pinball50_s"]:.3f} | '
                     f'{m["pinball90_s"]:.3f} | {m["coverage90"]:.2%} | {m["crossing_pct"]:.2f}% |')
    lines.append('')
    return lines


def evaluate_files(data: str | Path, queries: str | Path, truth: str | Path,
                   pred: str | Path, out: str | Path = 'eval/report.md') -> dict:
    root, qpath, tpath, ppath = map(Path, (data, queries, truth, pred))
    qdf = load_queries(qpath)
    tdf = _unique_csv(tpath, {'query_id', 'remaining_s'})
    pdf = _unique_csv(ppath, {'query_id', 'p50_s', 'p90_s'})
    if len(tdf) != len(qdf) or set(qdf.query_id) != set(tdf.query_id):
        raise ValueError('Truth query_id must exactly match queries')
    if not set(pdf.query_id).issubset(set(qdf.query_id)):
        raise ValueError('Predictions contain unknown query_id')

    base = _baseline_frame(root, qpath)
    pred_cols = pdf.rename(columns={'p50_s': 'p50_s_cb', 'p90_s': 'p90_s_cb'})
    df = (qdf.merge(tdf, on='query_id', how='left', validate='one_to_one')
          .merge(base, on='query_id', how='left', validate='one_to_one')
          .merge(pred_cols[['query_id', 'p50_s_cb', 'p90_s_cb']],
                 on='query_id', how='left', validate='one_to_one'))
    finite = np.isfinite(df[['p50_s_cb', 'p90_s_cb']].to_numpy(dtype=float)).all(axis=1)
    valid = finite & (df.p50_s_cb >= 0) & (df.p90_s_cb >= 0)
    failures = int((~valid).sum())
    df.loc[~valid, 'p50_s_cb'] = df.loc[~valid, 'p50_s_base']
    df.loc[~valid, 'p90_s_cb'] = df.loc[~valid, 'p90_s_base']
    df = load_context(root, df)
    lines = [
        '# ETA: проверка на отложенном периоде', '',
        f'Запросов: **{len(df):,}**; поездок: **{df.trip_id.nunique():,}**.', '',
        'Метрики считаются по одним и тем же запросам; при отсутствующем или некорректном '
        'предсказании подставляется baseline и запрос остаётся в метриках.', '',
        f'Предсказаний не было / были нечисловыми или отрицательными: **{failures}**. '
        'Детальные причины сбоев внутри модели не восстановимы из одного PRED.csv.', '',
    ]
    sidecar = ppath.with_suffix('.fallbacks.json')
    if sidecar.exists():
        try:
            info = json.loads(sidecar.read_text(encoding='utf-8'))
            lines += [f'Внутренние переходы к baseline во время predict: **{info.get("fallback_count", "неизвестно")}**. ',
                      f'Причины: `{json.dumps(info.get("reasons", {}), ensure_ascii=False)}`.', '']
        except (ValueError, OSError):
            lines += ['Файл fallback-диагностики обнаружен, но не прочитан.', '']
    else:
        lines += ['Внутренние fallback, уже записанные в PRED.csv, неотличимы от успешных прогнозов '
                  'без журнала предсказателя. Их число см. в выводе команды `make predict`.', '']

    lines += _table(df, 'Все запросы')
    slices = [
        ('Гружёные', df.loaded.astype(int) == 1),
        ('Пустые', df.loaded.astype(int) == 0),
        ('Дождь в момент прогноза', df.rain),
        ('Сухая погода', ~df.rain),
        ('Новые водители (по первому появлению)', df.new_driver),
        ('Остальные водители', ~df.new_driver),
    ]
    for bucket in ('<3 min', '3-10 min', '>10 min'):
        slices.append((f'Фактический остаток {bucket}', df.duration_bucket.astype(str) == bucket))
    for label, mask in slices:
        part = df.loc[mask]
        if not part.empty:
            lines += _table(part, label)

    # No explicit failure indicator exists in public telemetry. A conservative
    # proxy is diagnostic only and is NEVER presented as a true failure label.
    events = root / 'edge_events.csv'
    if events.exists():
        telemetry = pd.read_csv(events, usecols=['trip_id', 'enter_ts', 'exit_ts'])
        duration = telemetry.exit_ts - telemetry.enter_ts
        suspect_ids = set(telemetry.loc[duration >= 600, 'trip_id'])
        part = df.loc[df.trip_id.isin(suspect_ids)]
        if not part.empty:
            lines += _table(part, 'Подозрение на длительную остановку (прокси ≥600 с на ребре)')
    lines += [
        '## Ограничения интерпретации', '',
        '- В публичной телеметрии отсутствует метка поломки. Срез с длительным проездом '
        '**не равнозначен** срезу подтверждённых поломок.',
        '- Разбиение по фактическому оставшемуся времени использует истину **только для оценки**, не для признаков.',
        '- Запросы внутри одной поездки статистически зависимы; агрегированные метрики по запросам '
        'не следует считать независимыми наблюдениями.',
        '- Нового водителя определяем по дате первого появления в публичной истории поездок; '
        'это диагностическое правило, не часть инференса.',
        '- Временной holdout должен быть зафиксирован до выбора модели. Этот отчёт не является '
        'подтверждением, что выбор модели не использовал holdout.', '',
    ]
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')
    return {'n_queries': len(df), 'missing_or_invalid_predictions': failures,
            'mae_baseline_s': metrics(df, 'base')['mae_s'],
            'mae_model_s': metrics(df, 'cb')['mae_s'], 'report': str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--queries', required=True)
    parser.add_argument('--truth', required=True)
    parser.add_argument('--pred', required=True)
    parser.add_argument('--out', default='eval/report.md')
    args = parser.parse_args()
    print(json.dumps(evaluate_files(args.data, args.queries, args.truth, args.pred, args.out),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
