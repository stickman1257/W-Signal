import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy import stats

import argparse
_ap=argparse.ArgumentParser();_ap.add_argument('--source', type=Path, required=True);_ap.add_argument('--output', type=Path, required=True)
_args=_ap.parse_args();BASE=_args.source
RAW, PUB, FIG, TAB = BASE / 'origin', BASE / 'public', _args.output / 'figures', _args.output / 'tables'
FIG.mkdir(parents=True,exist_ok=True); TAB.mkdir(parents=True,exist_ok=True)

INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'
CRIT = '#d03b3b'
BLUES = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
SEQ = LinearSegmentedColormap.from_list('seq', BLUES)
DIV = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#86b6ef', '#f0efec', '#f08a8a', '#c43b3b'])
available_fonts = {font.name for font in font_manager.fontManager.ttflist}
font_family = next((name for name in ['Noto Sans CJK KR', 'Noto Sans KR', 'NanumGothic', 'Malgun Gothic', 'AppleGothic'] if name in available_fonts), 'DejaVu Sans')
plt.rcParams.update({
    'font.family': font_family, 'axes.unicode_minus': False, 'font.size': 10,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.titleweight': 'bold', 'axes.titlesize': 12, 'axes.titlecolor': INK,
    'axes.titlelocation': 'left', 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.dpi': 200,
    'legend.frameon': False, 'lines.linewidth': 2})

def save(f, name):
    f.tight_layout(); f.savefig(FIG / f'{name}.png', bbox_inches='tight'); plt.close(f)

A = pd.read_csv(RAW / 'water_signals_by_legal_dong.csv')
B = pd.read_csv(RAW / 'water_usage_by_district.csv')
K = pd.read_csv(RAW / 'income_by_admin_dong.csv', dtype={'행정동코드': str})
kik = pd.read_csv(PUB / 'daegu_kikmix_2024.csv', dtype=str)
geo = json.load(open(PUB / 'daegu_adm2024.geojson', encoding='utf-8'))

kik = kik[kik.읍면동명.notna() & kik.시군구명.notna()]
pairs = kik[['시군구명', '동리명', '행정동코드', '읍면동명']].drop_duplicates()
pairs = pairs.rename(columns={'시군구명': '시군구', '동리명': '읍면동', '읍면동명': '행정동'})
pairs['admin8'] = pairs.행정동코드.str[:8]
pairs = pairs.merge(A[['시군구', '읍면동']], on=['시군구', '읍면동'])
pairs['w'] = 1 / pairs.groupby(['시군구', '읍면동']).admin8.transform('nunique')
x = pairs.merge(A, on=['시군구', '읍면동'])
known = x.sig.notna()
x['wn'], x['wone'] = x.n * x.w, x.n_one.fillna(0) * x.w
x['wsig'] = (x.sig * x.w).where(known, 0)
x['wone_k'] = x.wone.where(known, 0)
D = x.groupby(['admin8', '시군구', '행정동'], as_index=False)[['wn', 'wone', 'wsig', 'wone_k']].sum()
D['n_legal'] = x.groupby('admin8').읍면동.nunique().reindex(D.admin8).values
D['split'] = x.groupby('admin8').w.min().reindex(D.admin8).lt(1).values
D['one_rate'] = D.wone / D.wn
D['sig_rate'] = (D.wsig / D.wone_k).where(D.wone_k >= 30)
D['coverage'] = D.wone_k / D.wone

K['admin8'] = K.행정동코드.str[:8]
kb = K[K.거주기준 == 1].pivot_table(index='admin8', columns='기준시점', values='low_rate')
kb2 = K[K.거주기준 == 2].pivot_table(index='admin8', columns='기준시점', values='low_rate')
D = D.merge(pd.DataFrame({'low_rate': kb[202312], 'low_rate_2020': kb[202003],
                          'low_rate_b2': kb2[202312]}).reset_index(), on='admin8', how='left')
D['low_change'] = D.low_rate - D.low_rate_2020

D['area'] = np.where(D.행정동.str.endswith('동'), '도시(동)', '농촌(읍·면)')
D['vul'] = D.low_rate
ok = D.sig_rate.notna() & D.vul.notna()
for a_ in D.area.unique():
    m_ = ok & (D.area == a_)
    D.loc[m_, 'sig_pct'] = D.loc[m_, 'sig_rate'].rank(pct=True)
    D.loc[m_, 'vul_pct'] = D.loc[m_, 'vul'].rank(pct=True)
    D.loc[m_, 'hs'] = D.loc[m_, 'sig_rate'].median()
    D.loc[m_, 'hx'] = D.loc[m_, 'vul'].median()
D['W'] = (D.sig_pct * D.vul_pct * 100).round(1)
D['quad'] = np.select([ok & (D.sig_rate >= D.hs) & (D.vul >= D.hx), ok & (D.sig_rate >= D.hs),
                       ok & (D.vul >= D.hx), ok], ['① 즉시 방문', '② 숨은 위험', '③ 구조 취약', '④ 관찰'], '산출 불가')
D['W_rank'] = D.groupby('area').W.rank(ascending=False, method='min')
U = ok & (D.area == '도시(동)')
hs, hx = D.loc[U, 'sig_rate'].median(), D.loc[U, 'vul'].median()
D.sort_values(['area', 'W'], ascending=[True, False]).drop(columns=['hs', 'hx']).round(4).to_csv(
    TAB / 'admin_signals.csv', index=False, encoding='utf-8-sig')

NEWS = {'복현1동': 27.8, '상인3동': 25.3, '월성2동': 23.5, '대명1동': 16.0, '송현2동': 16.0, '송현1동': 15.8,
        '범어2동': 15.6, '대명3동': 14.4, '신당동': 13.9, '대명9동': 13.7}
D['news_risk'] = D.행정동.map(NEWS)
V = D[D.news_risk.notna()][['시군구', '행정동', 'news_risk', 'sig_rate', 'vul', 'W', 'W_rank', 'quad', 'split']]
V.round(4).to_csv(TAB / 'news_area_comparison.csv', index=False, encoding='utf-8-sig')
n_ok = int(ok.sum())
val = dict(n_news=int(len(V)), n_scored=int(V.W.notna().sum()),
           mean_W_news=float(V.W.mean()), mean_W_all=float(D.W.mean()),
           mean_vul_pct_news=float(D.loc[D.news_risk.notna(), 'vul_pct'].mean()),
           mean_W_urban=float(D.loc[U, 'W'].mean()),
           mean_sig_pct_news=float(D.loc[D.news_risk.notna(), 'sig_pct'].mean()),
           in_q12=int(V.quad.isin(['① 즉시 방문', '③ 구조 취약']).sum()))
u = stats.mannwhitneyu(D.loc[D.news_risk.notna() & U, 'vul'], D.loc[D.news_risk.isna() & U, 'vul'], alternative='greater')
val['mwu_vul_p'] = float(u.pvalue)
u = stats.mannwhitneyu(D.loc[D.news_risk.notna() & U, 'W'], D.loc[D.news_risk.isna() & U, 'W'], alternative='greater')
val['mwu_W_p'] = float(u.pvalue)
u = stats.mannwhitneyu(D.loc[D.news_risk.notna() & U, 'sig_rate'], D.loc[D.news_risk.isna() & U, 'sig_rate'], alternative='greater')
val['mwu_sig_p'] = float(u.pvalue)
val['split_news'] = int(D.loc[D.news_risk.notna(), 'split'].sum())

B['ym'] = pd.to_datetime(B.납기년월.astype(str), format='%Y%m')
B['parity'] = np.where(B.ym.dt.month % 2 == 1, '홀수월', '짝수월')
main = B.groupby(['시군구', 'parity']).n.median().reset_index().sort_values('n').drop_duplicates('시군구', keep='last')
Bm = B.merge(main[['시군구', 'parity']], on=['시군구', 'parity'])
Bm = Bm[Bm.시군구 != '군위군']
base20 = Bm[Bm.ym.dt.year == 2020].groupby('시군구').day.mean()
Bm['idx'] = Bm.day / Bm.시군구.map(base20) * 100
Bm.to_csv(TAB / 'district_usage_index.csv', index=False, encoding='utf-8-sig')
chg = (Bm[Bm.ym >= '2024-01-01'].groupby('시군구').day.mean() /
       Bm[(Bm.ym >= '2020-01-01') & (Bm.ym < '2020-07-01')].groupby('시군구').day.mean() - 1) * 100

desc = pd.DataFrame({
    '법정동 신호율': A.rate.describe(), '법정동 1인추정비율': (A.n_one / A.n).describe(),
    '행정동 신호율': D.sig_rate.describe(), '행정동 저소득비율(KCB)': D.low_rate.describe(),
    '행정동 W-SIGNAL': D.W.describe()}).T.round(3)
desc.to_csv(TAB / 'descriptive_statistics.csv', encoding='utf-8-sig')
gu = D[ok].groupby(['area', '시군구']).agg(행정동수=('admin8', 'size'), 신호율_중앙값=('sig_rate', 'median'),
                                저소득비율_중앙값=('low_rate', 'median'), W_평균=('W', 'mean'),
                                즉시방문=('quad', lambda s: (s == '① 즉시 방문').sum()),
                                숨은위험=('quad', lambda s: (s == '② 숨은 위험').sum())).round(3)
gu.to_csv(TAB / 'district_summary.csv', encoding='utf-8-sig')
rho = {c: stats.spearmanr(D.loc[U, 'sig_rate'], D.loc[U, c], nan_policy='omit') for c in
       ['low_rate', 'one_rate', 'low_change', 'low_rate_b2']}
corr = pd.DataFrame({c: dict(rho=r.statistic, p=r.pvalue) for c, r in rho.items()}).T.round(4)
corr.to_csv(TAB / 'signal_correlations.csv', encoding='utf-8-sig')

f, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'width_ratios': [1, 1.6]})
ax[0].bar(['2023', '2024'], [3661, 3924], color=S1, width=.5)
for i, v in enumerate([3661, 3924]): ax[0].text(i, v + 60, f'{v:,}명', ha='center', color=INK, fontsize=11)
ax[0].set(title='전국 고독사 사망자', ylim=(0, 4500)); ax[0].grid(axis='x', visible=False)
grp = ['전체', '남성', '65세 이상', '무직', '원룸', '고시원', '쪽방·모텔', '수급자']
val_g = [7.6, 9.3, 9.6, 17.6, 15.2, 22.1, 24.5, 31.7]
ax[1].barh(grp[::-1], val_g[::-1], color=[S2 if g != '전체' else MUTED for g in grp[::-1]], height=.6)
for i, v in enumerate(val_g[::-1]): ax[1].text(v + .4, i, f'{v}%', va='center', color=INK, fontsize=9.5)
ax[1].set(title='대구 1인가구 고독사 고위험 비율 (2024 실태조사)', xlim=(0, 36)); ax[1].grid(axis='y', visible=False)
save(f, 'background')

gus = list(Bm.시군구.unique())
f, axs = plt.subplots(2, 4, figsize=(12, 5), sharex=True, sharey=True)
for a_, g in zip(axs.flat, gus):
    for o in gus:
        t = Bm[Bm.시군구 == o]; a_.plot(t.ym, t.idx, color=GRID, lw=1)
    t = Bm[Bm.시군구 == g]; a_.plot(t.ym, t.idx, color=S1)
    a_.axhline(100, color=AXIS, lw=.8); a_.set_title(f'{g}  {chg[g]:+.1f}%', fontsize=10.5)
for a_ in axs[:, 0]: a_.set_ylabel('2020년=100')
for a_ in axs.flat: a_.xaxis.set_major_locator(mdates.YearLocator()); a_.xaxis.set_major_formatter(mdates.DateFormatter('%y년'))
f.suptitle('구별 가정용 일평균 사용량 지수 (주 검침군, 2020~2024.6)  · 제목 옆 숫자 = 2024 상반기 vs 2020 상반기',
           x=.01, ha='left', fontsize=11.5, fontweight='bold', color=INK)
save(f, 'district_usage_trends')

f, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'width_ratios': [1, 1.5]})
ax[0].hist(A.rate.dropna() * 100, bins=18, color=S1, edgecolor='white', lw=2)
ax[0].axvline(A.rate.median() * 100, color=INK2, ls='--', lw=1)
ax[0].text(A.rate.median() * 100 + 1, ax[0].get_ylim()[1] * .9, f'중앙값 {A.rate.median() * 100:.1f}%', color=INK2)
ax[0].set(title='법정동별 신호 경험률 분포', xlabel='1인 추정 가구 중 최근 12개월 급감·장기0 경험 비율 (%)', ylabel='법정동 수')
order = A.groupby('시군구').rate.median().sort_values().index
data = [A.loc[A.시군구 == g, 'rate'].dropna() * 100 for g in order]
bp = ax[1].boxplot(data, vert=False, patch_artist=True, widths=.55, medianprops=dict(color=INK))
for p in bp['boxes']: p.set(facecolor='#cde2fb', edgecolor=S1)
ax[1].set_yticks(range(1, len(order) + 1), [f'{g} ({len(d)})' for g, d in zip(order, data)])
ax[1].set(title='구별 신호 경험률 (괄호 = 공개 법정동 수)', xlabel='%'); ax[1].grid(axis='y', visible=False)
save(f, 'signal_distribution')

f, ax = plt.subplots(figsize=(6.5, 4.4))
t = A.dropna(subset=['rate']).assign(one=lambda d: d.n_one / d.n)
ax.scatter(t.one * 100, t.rate * 100, s=np.clip(t.n / 25, 10, 160), color=S1, alpha=.55, edgecolor='white', lw=1)
r = stats.spearmanr(t.one, t.rate)
ax.set(title=f'1인 추정 비율이 높은 동일수록 신호율도 높다 (스피어만 ρ={r.statistic:.2f})',
       xlabel='1인 추정 가구 비율 (%)', ylabel='신호 경험률 (%)')
for _, rr in t.nlargest(5, 'rate').iterrows(): ax.annotate(f'{rr.시군구} {rr.읍면동}', (rr.one * 100, rr.rate * 100), fontsize=8, color=INK2, xytext=(4, 2), textcoords='offset points')
save(f, 'low_usage_signal_relationship')

def poly_list(feat):
    g = feat['geometry']; cs = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
    return [np.array(p[0]) for p in cs]

feats = [(f_['properties']['adm_cd2'][:8], poly_list(f_)) for f_ in geo['features']]

def choropleth(ax, values, cmap, norm, title, cbar_label, fmt='{:.0f}'):
    polys, cols, nan_polys = [], [], []
    for code, ps in feats:
        v = values.get(code, np.nan)
        for p in ps:
            (nan_polys if pd.isna(v) else polys).append(p)
            if not pd.isna(v): cols.append(cmap(norm(v)))
    ax.add_collection(PolyCollection(nan_polys, facecolor='#f0efec', edgecolor='white', lw=.4, hatch='////'))
    ax.add_collection(PolyCollection(polys, facecolor=cols, edgecolor='white', lw=.4))
    ax.autoscale(); ax.set_aspect(1.2); ax.axis('off'); ax.set_title(title)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = plt.colorbar(sm, ax=ax, fraction=.035, pad=.01); cb.set_label(cbar_label, color=INK2); cb.outline.set_visible(False)

V_sig = dict(zip(D.admin8, D.sig_rate * 100)); V_low = dict(zip(D.admin8, D.low_rate * 100)); V_W = dict(zip(D.admin8, D.W))
f, ax = plt.subplots(1, 2, figsize=(12, 6.4))
choropleth(ax[0], V_sig, SEQ, Normalize(10, 50), '행정동 신호 경험률 (상수도)', '%')
choropleth(ax[1], V_low, SEQ, Normalize(10, 50), '행정동 하위 소득구간 비율 (KCB 2023.4Q)', '%')
f.text(.01, .02, '빗금 = 비공개(1~4건) 또는 분모 부족으로 미산출', color=MUTED, fontsize=9)
save(f, 'signal_income_maps')
f, ax = plt.subplots(figsize=(7, 6.6))
choropleth(ax, V_W, SEQ, Normalize(0, 100), 'W-SIGNAL 지수 (0~100)', 'W-SIGNAL')
save(f, 'priority_map')

f, ax = plt.subplots(figsize=(7.6, 5.6))
q = D[U]
qc = {'① 즉시 방문': CRIT, '② 숨은 위험': S2, '③ 구조 취약': S1, '④ 관찰': AXIS}
for k_, c_ in qc.items():
    t = q[q.quad == k_]; ax.scatter(t.vul, t.sig_rate * 100, color=c_, s=38, edgecolor='white', lw=1.2, label=f'{k_} ({len(t)})', zorder=3)
ax.axvline(hx, color=INK2, lw=.8, ls='--'); ax.axhline(hs * 100, color=INK2, lw=.8, ls='--')
for i_, (_, rr) in enumerate(q.nlargest(7, 'W').iterrows()):
    ax.annotate(rr.행정동, (rr.vul, rr.sig_rate * 100), fontsize=8, color=INK, xytext=(5, 6 if i_ % 2 else -11), textcoords='offset points')
for _, rr in q[q.news_risk.notna()].iterrows():
    ax.scatter(rr.vul, rr.sig_rate * 100, s=110, facecolor='none', edgecolor=INK, lw=1.2, zorder=4)
ax.set(title='W-SIGNAL 4분면 · 도시 118개 동 (검은 원 = 언론 지목 고위험 동)', xlabel='구조 취약도 = 하위 소득구간 비율 (KCB)', ylabel='신호 경험률 (%)')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v*100:.0f}%'))
ax.legend(loc='upper left', fontsize=9)
save(f, 'care_groups')

t = D[U].nlargest(20, 'W').iloc[::-1]
f, ax = plt.subplots(figsize=(8, 6.2))
ax.barh(t.시군구 + ' ' + t.행정동, t.W, color=[qc[v] for v in t.quad], height=.65)
for i, (w_, qq) in enumerate(zip(t.W, t.quad)): ax.text(w_ + .8, i, f'{w_:.0f}', va='center', fontsize=9, color=INK)
ax.set(title='W-SIGNAL 상위 20개 행정동 (도시 동)', xlabel='W-SIGNAL (0~100)', xlim=(0, 108)); ax.grid(axis='y', visible=False)
save(f, 'top_priority_areas')

f, ax = plt.subplots(figsize=(8, 4.4))
t = D[D.news_risk.notna() & U].sort_values('news_risk')
yy = np.arange(len(t))
ax.barh(yy + .18, t.vul_pct * 100, height=.34, color=S1, label='구조 취약도(저소득) 백분위')
ax.barh(yy - .18, t.sig_pct * 100, height=.34, color=S2, label='신호 경험률 백분위')
ax.set_yticks(yy, [f'{a} ({b}명)' for a, b in zip(t.행정동, t.news_risk)])
ax.axvline(50, color=INK2, ls='--', lw=1)
ax.set(title='언론 지목 고위험 동의 순위 (괄호 = 인구 1천명당 고독사 위험군)', xlabel='도시 118개 동 중 백분위 (50 = 중간)', xlim=(0, 100))
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=2, fontsize=9); ax.grid(axis='y', visible=False)
save(f, 'external_comparison')

kt = K[K.거주기준 == 1].groupby('기준시점').low_rate.median()
f, ax = plt.subplots(figsize=(7.5, 3.4))
ax.plot([pd.Period(str(p)[:4] + 'Q' + str(int(str(p)[4:]) // 3), 'Q').to_timestamp() for p in kt.index], kt.values * 100, color=S1, marker='o', ms=4)
ax.set(title='대구 행정동 저소득 비율 중앙값 추이 (KCB, 거주기준 1)', ylabel='%')
save(f, 'income_trends')

json.dump(dict(validation=val, n_admin=int(len(D)), n_scored=n_ok, n_urban=int(U.sum()), n_rural=int((ok & ~U).sum()), sig_median=float(hs), vul_median=float(hx),
               n_legal=int(len(A)), n_legal_rate=int(A.rate.notna().sum()),
               legal_rate_median=float(A.rate.median()), legal_rate_mean=float(A.rate.mean()),
               one_rate_rho=float(r.statistic), quad_counts=D.quad.value_counts().to_dict(),
               split_admin=int(D.split.sum()), usage_change_2024_vs_2020=chg.round(2).to_dict(),
               corr=corr.to_dict()), open(TAB / 'summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.load(open(TAB / 'summary.json', encoding='utf-8')))
