from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut


def usage_results(source, output):
    b = pd.read_csv(source / 'data/origin/water_usage_by_district.csv')
    b = b[b.시군구.ne('군위군')].copy()
    b['ym'] = pd.to_datetime(b.납기년월.astype(str), format='%Y%m')
    b['series'] = b.시군구 + '_' + (b.ym.dt.month % 2).astype(str)
    b = b.sort_values(['series', 'ym']).reset_index(drop=True)
    for lag in range(1, 10):
        b[f'l{lag}'] = b.groupby('series').day.shift(lag)
    ratios = np.array([b.l1 / b.l7, b.l2 / b.l8, b.l3 / b.l9])
    b['pred'] = b.l6 * (ratios[0] + np.median(ratios, axis=0)) / 2
    b['baseline'] = b.l6 * ratios[0]
    valid = b.dropna(subset=['pred', 'baseline']).copy()
    calibration = valid[valid.ym.between('2023-01-01', '2023-06-30')]
    evaluation = valid[valid.ym.between('2023-07-01', '2024-06-30')].copy()
    errors = np.sort(np.abs(np.log(calibration.day / calibration.pred)).to_numpy())
    quantiles = {}
    for level in [90, 95]:
        q = float(errors[min(len(errors)-1, math.ceil((len(errors)+1)*level/100)-1)])
        quantiles[level] = q
        evaluation[f'lo{level}'] = evaluation.pred * np.exp(-q)
        evaluation[f'hi{level}'] = evaluation.pred * np.exp(q)
        evaluation[f'covered{level}'] = evaluation.day.between(evaluation[f'lo{level}'], evaluation[f'hi{level}'])
    evaluation = evaluation.rename(columns={'시군구': 'gu', 'day': 'actual'})
    evaluation['train_end'] = (evaluation.ym - pd.DateOffset(months=2)).dt.strftime('%Y-%m')
    evaluation['ym'] = evaluation.ym.dt.strftime('%Y-%m')
    evaluation['model'] = 'seasonal_ratio_blend'
    columns = ['ym', 'gu', 'series', 'actual', 'model', 'pred', 'train_end',
               'lo90', 'hi90', 'covered90', 'lo95', 'hi95', 'covered95', 'baseline']
    evaluation[columns].to_csv(output / 'usage_selected_predictions.csv', index=False, encoding='utf-8-sig')
    loss = np.abs(evaluation.actual - evaluation.pred)
    baseline_mape = float(np.mean(np.abs(evaluation.actual - evaluation.baseline) / evaluation.actual)*100)
    mape = float(np.mean(loss / evaluation.actual)*100)
    return {'n': len(evaluation), 'calibration_n': len(calibration), 'MAPE': mape,
            'MAE': float(loss.mean()), 'WAPE': float(loss.sum()/evaluation.actual.sum()*100),
            'baseline_MAPE': baseline_mape, 'relative_MAPE_reduction_pct': (1-mape/baseline_mape)*100,
            'q90_log': quantiles[90], 'q95_log': quantiles[95], 'coverage90': float(evaluation.covered90.mean()),
            'coverage95': float(evaluation.covered95.mean()),
            'mean_width90_pct': float(((evaluation.hi90-evaluation.lo90)/evaluation.pred).mean()*100),
            'scope': '동일 검침군의 다음 고지 예측. 개발 과정에서 확인한 과거 구간의 평가.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('outputs/reproduction'))
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    raw = source / 'data/origin'
    for name in ['water_signals_by_legal_dong.csv', 'water_usage_by_district.csv', 'income_by_admin_dong.csv']:
        if not (raw/name).exists():
            raise FileNotFoundError(f'승인 집계 배치가 필요합니다: {raw/name}. data/origin/README.md를 확인하세요.')
    output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('model_source', source/'src/experiment.py')
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    model.SRC, model.OUT = source/'data', output
    x = model.signal_data()
    sets = {'수도·규모·지역': ['one_rate', 'log_n', 'rural'],
            '소득·지역': ['low', 'low_b2', 'low_change', 'rural'],
            '수도+소득·지역': ['one_rate', 'log_n', 'low', 'low_b2', 'low_change', 'rural']}
    rows = []
    for fold, (train, test) in enumerate(LeaveOneGroupOut().split(x, groups=x.시군구)):
        for name, columns in sets.items():
            model.NUM = columns
            pred = model.predict_signal('blend_old', x.iloc[train], x.iloc[test])
            for index, value in zip(test, pred):
                rows.append({'fold': fold, 'gu': x.iloc[index].시군구, 'legal': x.iloc[index].읍면동,
                             'feature_set': name, 'actual': float(x.iloc[index].rate), 'pred': float(value)})
        print('completed outer fold', fold, flush=True)
    predictions = pd.DataFrame(rows)
    predictions.to_csv(output/'feature_ablation_predictions.csv', index=False, encoding='utf-8-sig')
    metrics = [{'feature_set': name, 'n': len(frame),
                'MAE_pp': mean_absolute_error(frame.actual, frame.pred)*100,
                'R2': r2_score(frame.actual, frame.pred)} for name, frame in predictions.groupby('feature_set')]
    summary = {'protocol': 'Ridge·CatBoost 50:50 고정 모형, 구군 분리9회, 입력 묶음별 튜닝 없음.',
               'interpretation': '지역 신호율 회귀와 집계 사용량 예측. 가구별 발견·예방 효과는 현장 평가 대상.',
               'source_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(raw.glob('*.csv'))},
               'signal_input_ablation': metrics, 'usage_recomputed': usage_results(source, output),
               'model_settings': {'ridge': 'median + StandardScaler + OHE gu + RidgeCV logspace(-3,3,13)',
                                  'catboost': {'iterations': 350, 'depth': 3, 'learning_rate': .03,
                                               'l2_leaf_reg': 8, 'seed': 42}, 'blend': [.5, .5]}}
    (output/'reproduction_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
