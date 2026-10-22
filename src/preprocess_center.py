# 센터 원본으로 재실행 필요. 공개 집계만으로 원본 전처리 재현을 확인할 수 없습니다.
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

METER = r"../import_data/TB_CKNU_DGWATER_METER_INFORMATION/TB_DGWATER_MeterInformation_UTF.csv"
NOTICE = sorted(glob.glob(r"../import_data/TB_CKNU_DGWATER_BILLING_INFORMATION/TB_DGWATER_NoticeInformation_202001~202406/*.csv"))
KCB = r'../import_data/TB_KCB_INCOME_STAT_DATASET/TB_KCB_INCOME_STAT_DATASET.csv'
Path('export').mkdir(exist_ok=True)
if not NOTICE:
    raise FileNotFoundError('센터 고지 원본 경로를 확인해주세요.')

print('clear')

def rd(f, c):
    try:
        return pd.read_csv(f, dtype=str, usecols=c, encoding='utf-8-sig')
    except UnicodeDecodeError:
        return pd.read_csv(f, dtype=str, usecols=c, encoding='cp949')

m = rd(METER, ['고객번호', '계량기구분', '시군구', '읍면동', '상수도주업종', '급수전상태'])
m = m[m.상수도주업종.str.contains('가정', na=False) & ~m.급수전상태.str.contains('중지|폐', na=False)]
m['k'] = m.고객번호 + '|' + m.계량기구분.fillna('')
m = m.drop_duplicates('k').set_index('k')

c = ['고객번호', '계량기구분', '납기년월', '전월검침일자', '당월검침일자', '사용량', '격월고지여부']
n = pd.concat([rd(f, c) for f in NOTICE])
n['k'] = n.고객번호 + '|' + n.계량기구분.fillna('')
n = n[n.k.isin(m.index)].drop_duplicates(['k', '납기년월']).sort_values(['k', '납기년월']).reset_index(drop=True)
d = (pd.to_datetime(n.당월검침일자, format='%Y%m%d', errors='coerce') - pd.to_datetime(n.전월검침일자, format='%Y%m%d', errors='coerce')).dt.days
d = d.where(d.between(20, 70), np.where(n.격월고지여부 == 'Y', 60, 30))
n['u'] = pd.to_numeric(n.사용량, errors='coerce')
n['day'] = n.u / d

print('clear')

g = n.groupby('k')
n['base'] = g.day.shift().groupby(n.k).rolling(6,min_periods=3).median().droplevel(0)
lo, z = n.day < n.base * 0.3, n.u == 0
n['S1'] = lo & lo.groupby(n.k).shift(fill_value=False)
n['S2'] = z & z.groupby(n.k).shift(fill_value=False)
med = n[n.day > 0].groupby('k').day.median()

B = n.join(m[['시군구']], on='k').groupby(['시군구', '납기년월']).agg(n=('day', 'size'), day=('day', 'median')).reset_index()
B[B.n >= 30].round(4).to_csv('export/water_usage_by_district.csv', index=False, encoding='utf-8-sig')

print('clear')

n = n[n.납기년월 > str(int(n.납기년월.max()) - 100)]
k = n.groupby('k')[['S1', 'S2']].any().join(med.rename('med')).join(m[['시군구', '읍면동']])
k['one'] = k.med <= 0.2
k['sig'] = k.one & (k.S1 | k.S2)
A = k.groupby(['시군구', '읍면동']).agg(n=('one', 'size'), n_one=('one', 'sum'), sig=('sig', 'sum')).reset_index()
A = A[A.n >= 30]
A['rate'] = A.sig / A.n_one
A.loc[A.sig.between(1, 4) | (A.n_one < 30), ['sig', 'rate']] = np.nan
A.loc[A.n_one.between(1, 4), 'n_one'] = np.nan
A.round(4).to_csv('export/water_signals_by_legal_dong.csv', index=False, encoding='utf-8-sig')

print('clear')

K = pd.read_csv(KCB, dtype=str, encoding='utf-8-sig')
K = K[K.행정동코드.str.startswith('27')]
b = [x for x in K.columns if x.startswith('월환산소득')]
if len(b) < 3:
    raise ValueError('KCB 소득구간 컬럼을 확인해주세요.')
# b[:3]의 실제 구간이 월 200만원 이하에 해당하는지 원본 명세와 대조 후 실행합니다.
K[b]=K[b].apply(pd.to_numeric, errors='coerce')
K = K.groupby(['기준시점', '거주기준', '행정동코드'])[b].sum()
K = pd.DataFrame({'total': K.sum(axis=1), 'low': K[b[:3]].sum(axis=1)}).reset_index()
K['low_rate'] = K.low / K.total

K.loc[K.low.between(1, 4), 'low_rate'] = np.nan
K = K.loc[K.total >= 30].drop(columns=['total', 'low'])
K.round(4).to_csv('export/income_by_admin_dong.csv', index=False, encoding='utf-8-sig')
print(list(Path('export').iterdir()))

print('clear')
