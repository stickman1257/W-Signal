from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def run(source, output, prediction_file=None):
    output.mkdir(parents=True, exist_ok=True)
    raw = source / 'data/origin'
    a = pd.read_csv(raw / 'water_signals_by_legal_dong.csv')
    k = pd.read_csv(raw / 'income_by_admin_dong.csv', dtype={'행정동코드': str})
    link = pd.read_csv(source / 'data/public/daegu_kikmix_2024.csv', dtype=str)
    link = link[link.읍면동명.notna() & link.시군구명.notna()]
    pairs = link[['시군구명', '동리명', '행정동코드', '읍면동명']].drop_duplicates()
    pairs = pairs.rename(columns={'시군구명': '시군구', '동리명': '읍면동', '읍면동명': '행정동'})
    pairs['admin8'] = pairs.행정동코드.str[:8]
    x = pairs.merge(a, on=['시군구', '읍면동'], validate='many_to_one')
    x['weight'] = 1 / x.groupby(['시군구', '읍면동']).admin8.transform('nunique')
    x['wn'] = x.n * x.weight
    x['wone'] = x.n_one.fillna(0) * x.weight
    x['wsig'] = (x.sig * x.weight).where(x.sig.notna(), 0)
    x['wone_k'] = x.wone.where(x.sig.notna(), 0)
    d = x.groupby(['admin8', '시군구', '행정동'], as_index=False)[['wn', 'wone', 'wsig', 'wone_k']].sum()
    d['split'] = x.groupby('admin8').weight.min().reindex(d.admin8).lt(1).to_numpy()
    d['coverage'] = d.wone_k / d.wone
    d['sig_rate'] = (d.wsig / d.wone_k).where(d.wone_k >= 30)
    k['admin8'] = k.행정동코드.str[:8]
    income = k[k.거주기준.eq(1) & k.기준시점.eq(202312)][['admin8', 'low_rate']]
    d = d.merge(income, on='admin8', how='left', validate='one_to_one')
    d['area'] = np.where(d.행정동.str.endswith('동'), 'urban', 'rural')
    d['scored'] = d.sig_rate.notna() & d.low_rate.notna()
    for _, ix in d[d.scored].groupby('area').groups.items():
        d.loc[ix, 'signal_percentile'] = d.loc[ix, 'sig_rate'].rank(pct=True)
        d.loc[ix, 'income_percentile'] = d.loc[ix, 'low_rate'].rank(pct=True)
    d['W'] = 100 * d.signal_percentile * d.income_percentile
    d['coverage_status'] = np.select(
        [d.coverage.ge(1 - 1e-9), d.coverage.gt(0)], ['full', 'partial'], default='unavailable')
    d['linkage_status'] = np.where(d.split, 'equal_allocation', 'unsplit_aggregate_link')
    d.to_csv(output / 'spatial_quality.csv', index=False, encoding='utf-8-sig')
    discrepancy = None
    supplied_file = source / 'data/derived/admin_signals.csv'
    if supplied_file.exists():
        supplied = pd.read_csv(supplied_file, dtype={'admin8': str})
        check = d.merge(supplied[['admin8', 'W']], on='admin8', suffixes=('', '_supplied'))
        discrepancy = float((check.W.round(1) - check.W_supplied).abs().max())
        if discrepancy > 0.100001:
            raise ValueError(f'지수 대조 결과를 확인하세요: {discrepancy}')
    pilot = d[d.scored & d.area.eq('urban') & ~d['split'] & d.coverage.ge(1-1e-9)].copy()
    pilot.to_csv(output / 'pilot_eligibility.csv', index=False, encoding='utf-8-sig')
    pilot_summary = {'scored_urban': int((d.scored & d.area.eq('urban')).sum()),
                     'unsplit_urban': int((d.scored & d.area.eq('urban') & ~d['split']).sum()),
                     'unsplit_full_coverage_urban': len(pilot),
                     'conditions': '도시 점수 제공·공간 배분 없음·공개 분모 전체 포함',
                     'interpretation': '집계 품질에 따른 시범 업무 검토 범위. 실제 위기가구 여부와 별도.'}
    (output / 'pilot_summary.json').write_text(json.dumps(pilot_summary, ensure_ascii=False, indent=2))

    u = d[d.scored & d.area.eq('urban')].copy()
    scores = {'product': u.W}
    for weight in [.25, .5, .75]:
        scores[f'sum_signal_{weight:.2f}'] = 100 * (
            weight * u.signal_percentile + (1 - weight) * u.income_percentile)
        scores[f'geometric_signal_{weight:.2f}'] = 100 * (
            u.signal_percentile ** weight * u.income_percentile ** (1 - weight))
    scores['minimum'] = 100 * np.minimum(u.signal_percentile, u.income_percentile)
    scores['signal_only'] = u.sig_rate
    scores['income_only'] = u.low_rate

    def top(values):
        q = u.assign(candidate=values).sort_values(['candidate', 'admin8'], ascending=[False, True])
        return set(q.head(20).admin8)

    baseline = top(scores['product'])
    sensitivity = []
    memberships = []
    for name, values in scores.items():
        chosen = top(values)
        sensitivity.append({'candidate': name, 'n': len(u), 'top_k': 20,
                            'overlap_with_product': len(baseline & chosen),
                            'rank_rho_with_product': spearmanr(u.W, values).statistic})
        rank = values.rank(ascending=False, method='min')
        for ix, row in u.iterrows():
            memberships.append({'candidate': name, 'admin8': row.admin8, 'district': row.시군구,
                                'admin_dong': row.행정동, 'rank': rank.loc[ix],
                                'in_top20': row.admin8 in chosen})
    pd.DataFrame(sensitivity).to_csv(output / 'priority_sensitivity.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(memberships).to_csv(output / 'priority_rankings.csv', index=False, encoding='utf-8-sig')
    candidate_names = [name for name in scores if name not in ['signal_only', 'income_only']]
    consensus = set.intersection(*(top(scores[name]) for name in candidate_names))
    signal_top = top(scores['signal_only'])
    queue_union = baseline | signal_top
    dual = u[u.admin8.isin(queue_union)].copy()
    dual['priority_queue'] = dual.admin8.isin(baseline)
    dual['signal_queue'] = dual.admin8.isin(signal_top)
    dual['stable_top20_across_formulas'] = dual.admin8.isin(consensus)
    dual.to_csv(output / 'dual_review_queues.csv', index=False, encoding='utf-8-sig')

    legal_income = x[['시군구', '읍면동', 'admin8']].drop_duplicates().merge(income, on='admin8')
    legal_income = legal_income.groupby(['시군구', '읍면동'], as_index=False).low_rate.mean()
    legal = a.merge(legal_income, on=['시군구', '읍면동'], how='left').dropna(subset=['rate', 'low_rate'])
    legal['one_rate'] = legal.n_one / legal.n
    correlations = {}
    for name, values in [('income', legal.low_rate), ('low_usage_share', legal.one_rate)]:
        result = spearmanr(legal.rate, values)
        correlations[name] = {'n_legal_dong': len(legal), 'rho': result.statistic,
                              'p_descriptive': result.pvalue}
    legal.to_csv(output / 'legal_dong_correlation_data.csv', index=False, encoding='utf-8-sig')

    predictions = pd.read_csv(prediction_file or source / 'results/feature_ablation_predictions.csv')
    w = predictions[predictions.feature_set.eq('수도·규모·지역')]
    all_inputs = predictions[predictions.feature_set.eq('수도+소득·지역')]
    paired = w.merge(all_inputs, on=['fold', 'gu', 'legal'], suffixes=('_water', '_all'), validate='one_to_one')
    paired['loss_difference_pp'] = 100 * (
        (paired.actual_all - paired.pred_all).abs() - (paired.actual_water - paired.pred_water).abs())
    groups = [v.loss_difference_pp.to_numpy() for _, v in paired.groupby('gu')]
    rng = np.random.default_rng(42)
    boot = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
            for _ in range(5000)]
    by_group = paired.groupby('gu').agg(n=('legal', 'size'), income_minus_water_MAE_pp=('loss_difference_pp', 'mean'))
    by_group.to_csv(output / 'income_effect_by_district.csv', encoding='utf-8-sig')
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(raw.glob('*.csv'))}
    summary = {
        'scope': '승인 반출 집계와 제공 연결표의 지역 분석. 가구 발견·예방 효과는 후속 현장 평가 대상.',
        'source_sha256': manifest,
        'spatial': {
            'admin_dong_total': len(d), 'scored': int(d.scored.sum()),
            'equal_allocation': int(d.split.sum()), 'unsplit_aggregate_link': int((~d.split).sum()),
            'full_coverage': int(d.coverage.ge(1 - 1e-9).sum()),
            'partial_coverage': int((d.coverage.gt(0) & d.coverage.lt(1 - 1e-9)).sum()),
            'no_public_denominator': int((~d.coverage.gt(0)).sum()),
            'scored_full_coverage': int((d.scored & d.coverage.ge(1 - 1e-9)).sum()),
            'scored_partial_coverage': int((d.scored & d.coverage.gt(0) & d.coverage.lt(1 - 1e-9)).sum()),
            'scored_equal_allocation': int((d.scored & d.split).sum()),
            'quality_cross_tab': d.groupby(['linkage_status', 'coverage_status']).size().to_dict(),
            'existing_W_max_rounding_difference': discrepancy,
            'interpretation': '분할 없는 지역 연결도 가구 주소 직접 연결을 의미하지 않음. 공개 분모 포함률은 저사용 계량기 분모 기준.',
        },
        'priority': {
            'urban_n': len(u), 'top_k': 20, 'formulas': sensitivity,
            'stable_top20_count': len(consensus),
            'stable_admin_dong': u[u.admin8.isin(consensus)].행정동.tolist(),
            'W_signal_top20_overlap': len(baseline & signal_top),
            'dual_queue_union': len(queue_union),
            'interpretation': '상위20은 비교용 고정 검토량 가정. 순위 겹침은 발견 정확도가 아님. 두 목록은 각각의 관점으로 병행 조회하며 합집합 전체를 방문하도록 요구하지 않음.',
        },
        'legal_dong_correlations': correlations,
        'income_prediction_effect': {
            'n': len(paired), 'district_blocks': len(groups),
            'all_minus_water_MAE_pp': paired.loss_difference_pp.mean(),
            'cluster_bootstrap_95_interval_pp': np.quantile(boot, [.025, .975]).tolist(),
            'bootstrap_repeats': 5000, 'seed': 42,
            'interpretation': '양수는 소득 포함 모형의 오차 증가. 구군 블록 수가 작고 고정 모형의 입력 비교이므로 일반적인 소득 무용론이나 통계적 우월성 판단으로 확대하지 않음.',
        },
    }
    (output / 'priority_audit.json').write_text(json.dumps(clean(summary), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(clean(summary), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--predictions', type=Path)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve(), args.predictions.resolve() if args.predictions else None)
