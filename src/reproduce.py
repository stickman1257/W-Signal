from pathlib import Path
import importlib.util,json,hashlib,argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error,r2_score

ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser()
ap.add_argument('--source',type=Path,default=ROOT,help='data·src·results가 있는 제출본 폴더')
ap.add_argument('--output',type=Path,default=ROOT/'outputs/reproduction')
args=ap.parse_args()
BASE=args.source.resolve()
OUT=args.output.resolve()
OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('model_source',BASE/'src/experiment.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
x=m.signal_data();rows=[]
sets={'수도·규모·지역':['one_rate','log_n','rural'],
      '소득·지역':['low','low_b2','low_change','rural'],
      '수도+소득·지역':['one_rate','log_n','low','low_b2','low_change','rural']}
for fold,(it,iv) in enumerate(LeaveOneGroupOut().split(x,groups=x.시군구)):
 for name,cols in sets.items():
  m.NUM=cols
  p=m.predict_signal('blend_old',x.iloc[it],x.iloc[iv])
  for idx,v in zip(iv,p):
   rows.append({'fold':fold,'gu':x.iloc[idx].시군구,'legal':x.iloc[idx].읍면동,
    'feature_set':name,'actual':float(x.iloc[idx].rate),'pred':float(v)})
 print('completed outer fold',fold,flush=True)
p=pd.DataFrame(rows);p.to_csv(OUT/'feature_ablation_predictions.csv',index=False,encoding='utf-8-sig')
metrics=[]
for n,t in p.groupby('feature_set'):
 metrics.append({'feature_set':n,'n':len(t),'MAE_pp':mean_absolute_error(t.actual,t.pred)*100,
 'R2':r2_score(t.actual,t.pred)})
u=pd.read_csv(BASE/'results/usage_selected_predictions.csv')
print('usage fields',list(u),flush=True)
actual=u.actual.to_numpy();pred=u.pred.to_numpy()
usage={'n':len(u),'MAPE':float(np.mean(np.abs(actual-pred)/actual)*100),
 'MAE':float(np.mean(np.abs(actual-pred))),'WAPE':float(np.abs(actual-pred).sum()/actual.sum()*100)}

b=pd.read_csv(BASE/'data/origin/water_usage_by_district.csv')
b=b[b.시군구.ne('군위군')].copy()
b['ym']=pd.to_datetime(b.납기년월.astype(str),format='%Y%m').dt.strftime('%Y-%m')
b['series']=b.시군구+'_'+(pd.to_datetime(b.ym).dt.month%2).astype(str)
b=b.sort_values(['series','ym']);g=b.groupby('series').day
for k in range(1,10):b['l'+str(k)]=g.shift(k)
b['formula']=b.l6*(b.l1/b.l7+np.median(np.array([b.l1/b.l7,b.l2/b.l8,b.l3/b.l9]),axis=0))/2
chk=u.merge(b[['series','ym','formula']],on=['series','ym'],validate='one_to_one')
usage['formula_max_abs_difference']=float(np.max(np.abs(chk.pred-chk.formula)))
summary={'protocol':'Ridge·CatBoost 50:50 고정 모형, 구·군 분리 9회, 입력 묶음별 추가 튜닝 없음.',
 'interpretation':'지역 신호율의 추정 성능이며, 가구별 고립·사망 예측 성능과 구분함.',
 'signal_input_ablation':metrics,'usage_recomputed':usage,
 'model_settings':{'ridge':'median imputation + StandardScaler + OHE gu + RidgeCV alphas 10^-3..10^3 (13)',
 'catboost':{'iterations':350,'depth':3,'learning_rate':0.03,'l2_leaf_reg':8,'seed':42},'blend':[0.5,0.5]}}

summary['protocol']='동일한 Ridge·CatBoost 50:50 모형, 구·군 분리 9회. 각 입력 묶음에 맞춘 추가 튜닝은 수행하지 않음.'
summary['interpretation']='지역 신호율 추정 성능이며, 개별 가구의 고립 또는 고독사 예측 성능과 구분함.'
(OUT/'reproduction_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
