import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate(frame, capacity, policy, share=0.6):
    reference = set(frame.sort_values(['sig_rate', 'admin8'], ascending=[False, True]).head(capacity).admin8)
    if policy == 'signal_only':
        selected = frame.sort_values(['sig_rate', 'admin8'], ascending=[False, True]).head(capacity)
    elif policy == 'index_only':
        selected = frame.sort_values(['W', 'admin8'], ascending=[False, True]).head(capacity)
    else:
        signal_high = frame.sig_rate >= frame.sig_rate.median()
        income_high = frame.low_rate >= frame.low_rate.median()
        a = frame[signal_high & income_high].sort_values(['W', 'admin8'], ascending=[False, True])
        b = frame[signal_high & ~income_high].sort_values(['sig_rate', 'admin8'], ascending=[False, True])
        a_count = int(np.ceil(capacity * share))
        b_count = capacity - a_count
        selected = pd.concat([a.head(a_count), b.head(b_count)])
        remaining = pd.concat([a.iloc[a_count:], b.iloc[b_count:]])
        selected = pd.concat([selected, remaining.head(max(0, capacity - len(selected)))])
    result = {'capacity': capacity, 'policy': policy, 'priority_share': share if policy == 'dual_queue' else None,
              'selected': len(selected), 'high_signal_top_k_overlap': len(reference & set(selected.admin8)),
              'mean_income_percentile': float(selected.income_percentile.mean()),
              'mean_signal_rate_pct': float(selected.sig_rate.mean() * 100),
              'allocated_count': int(selected['split'].sum()),
              'partial_coverage_count': int(selected.coverage.lt(0.9999).sum())}
    rows = selected[['admin8', '시군구', '행정동', 'W', 'sig_rate', 'low_rate', 'income_percentile', 'split', 'coverage']].copy()
    rows.insert(0, 'rank_in_policy', np.arange(1, len(rows) + 1))
    for key in ['capacity', 'policy', 'priority_share']:
        rows[key] = result[key]
    return result, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('results/spatial_quality.csv'))
    parser.add_argument('--output', type=Path, default=Path('outputs/review_policy'))
    args = parser.parse_args()
    frame = pd.read_csv(args.input, dtype={'admin8': str})
    urban = frame[frame.scored & frame.area.eq('urban')].copy()
    assert len(urban) == 118 and urban.admin8.nunique() == 118
    results, selections = [], []
    for capacity in [10, 20, 30]:
        for policy, share in [('signal_only', .6), ('index_only', .6)] + [('dual_queue', x) for x in [.25, .5, .6, .75]]:
            result, selected = evaluate(urban, capacity, policy, share)
            results.append(result)
            selections.append(selected)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output / 'review_policy_metrics.csv', index=False)
    pd.DataFrame([row for frame in selections for row in frame.to_dict('records')]).to_csv(args.output / 'review_policy_selections.csv', index=False)
    summary = {'population': '도시 점수 제공 행정동 118개', 'unit': '행정동',
               'capacity': '10·20·30개 지역의 고정 검토량 가정',
               'dual_queue': '수도신호와 소득이 모두 중앙값 이상인 목록을 W 순, 수도신호만 중앙값 이상인 목록을 신호율 순으로 배분. 부족분은 남은 두 목록 순서로 충원.',
               'default_share': .6, 'share_basis': '업무 배분 가정. 최적값 추정이나 결과에 맞춘 튜닝으로 해석하지 않음.',
               'ties': '지표 내림차순, 행정동 코드 오름차순',
               'interpretation': '지역 선택의 관점과 교환관계를 비교한 결정규칙 시뮬레이션. 실제 위기가구 재현율·지원 효과·담당자 시간 절감의 측정값과 구분.',
               'metrics_at_20': [x for x in results if x['capacity'] == 20],
               'quality_policy': '배분·부분공개 여부를 선택 결과와 함께 제공. 미산출 15개 동은 별도 집계 확인 목록으로 관리.'}
    (args.output / 'review_policy_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(pd.DataFrame(summary['metrics_at_20']).to_string(index=False))


if __name__ == '__main__':
    main()
