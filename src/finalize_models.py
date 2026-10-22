from pathlib import Path
import json,math
import pandas as pd,numpy as np
from experiment import usage_data,signal_data,predict_signal,dump,SRC,OUT

u=json.loads((OUT/'usage_experiments.json').read_text());s=json.loads((OUT/'signal_experiments.json').read_text())
p=pd.read_csv(OUT/'usage_expanding_predictions.csv');pick=u['frozen_2022_choice'];t=p[(p.model==pick)&(p.ym>='2023-07')].copy()
for l in [.90,.95]:
 q=u['frozen_choice_intervals'][str(l)]['qlog'];tag=str(int(l*100));t['lo'+tag]=t.pred*np.exp(-q);t['hi'+tag]=t.pred*np.exp(q)
 t['covered'+tag]=(t.actual>=t['lo'+tag])&(t.actual<=t['hi'+tag])
t.to_csv(OUT/'usage_selected_predictions.csv',index=False,encoding='utf-8-sig')
old=p[(p.model=='trend')&(p.ym>='2023-07')][['ym','gu','pred']].rename(columns={'pred':'old'})
z=t.merge(old,on=['ym','gu'],validate='1:1');z['quarter']=pd.to_datetime(z.ym).dt.to_period('Q').astype(str)
quarter=[]
for q,g in z.groupby('quarter'):
 quarter.append({'quarter':q,'n':len(g),'old_MAPE':np.mean(np.abs(g.actual-g.old)/g.actual)*100,'new_MAPE':np.mean(np.abs(g.actual-g.pred)/g.actual)*100})
pd.DataFrame(quarter).to_csv(OUT/'usage_quarterly_comparison.csv',index=False,encoding='utf-8-sig')
gstats=[]
for gu,g in t.groupby('gu'):
 gstats.append({'gu':gu,'n':len(g),'MAPE':np.mean(np.abs(g.actual-g.pred)/g.actual)*100,'MAE':np.mean(np.abs(g.actual-g.pred)),'coverage90':g.covered90.mean()})
pd.DataFrame(gstats).to_csv(OUT/'usage_district_metrics.csv',index=False,encoding='utf-8-sig')

blocks=[g for _,g in z.groupby('gu')];rng=np.random.default_rng(42);loss=[]
for _ in range(2000):
 a=pd.concat([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))]);loss.append((np.abs(a.actual-a.pred)/a.actual-np.abs(a.actual-a.old)/a.actual).mean()*100)
u['paired_new_minus_old_MAPE_CI95']=np.quantile(loss,[.025,.975]).tolist();u['quarter_metrics']=quarter;u['by_gu']=gstats

sp=pd.read_csv(OUT/'signal_predictions.csv');z=sp[sp.model.eq('blend_svr')].merge(sp[sp.model.eq('blend_old')][['id','pred']].rename(columns={'pred':'old'}),on='id',validate='1:1')
blocks=[g for _,g in z.groupby('gu')];diff=[]
for _ in range(2000):
 a=pd.concat([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))]);diff.append((np.abs(a.actual-a.pred)-np.abs(a.actual-a.old)).mean()*100)
s['exploratory_svr_minus_previous_CI95']=np.quantile(diff,[.025,.975]).tolist();s['adoption']='기존 Ridge CatBoost 50대50 유지'
dump(s,OUT/'signal_experiments.json');dump(u,OUT/'usage_experiments.json')

b=pd.read_csv(SRC/'origin/water_usage_by_district.csv');b=b[b.시군구.ne('군위군')].copy();b['ym']=pd.to_datetime(b.납기년월.astype(str),format='%Y%m');b['parity']=b.ym.dt.month%2;b['series']=b.시군구+'_'+b.parity.astype(str)
hist=b.sort_values(['series','ym']).copy();future=[]
for month in pd.date_range('2024-07-01','2024-12-01',freq='MS'):
 for se in hist[hist.parity.eq(month.month%2)].series.unique():
  h=hist[hist.series.eq(se)].sort_values('ym');v=h.day.to_numpy();ratio=[v[-k]/v[-k-6] for k in [1,2,3]]
  pred=v[-6]*(ratio[0]+np.median(ratio))/2
  horizon=int((h.ym>='2024-07-01').sum())+1
  future.append({'gu':h.시군구.iloc[-1],'series':se,'ym':month.strftime('%Y-%m'),'prediction':pred,'horizon':horizon})
  hist=pd.concat([hist,pd.DataFrame([{'시군구':h.시군구.iloc[-1],'series':se,'parity':month.month%2,'ym':month,'day':pred}])],ignore_index=True)
pd.DataFrame(future).to_csv(OUT/'usage_recursive_forecast.csv',index=False,encoding='utf-8-sig')
dump({'model':'직전 추세와 3회 중앙값 추세의 동일가중 결합','selection_year':'2022','weights':[.5,.5],
 'formula':'lag6 * (lag1/lag7 + median(lag1/lag7, lag2/lag8, lag3/lag9)) / 2',
 'same_parity_bills':True,'calibration':'2023-01~06','evaluation':'2023-07~2024-06','evaluation_status':'개발 과정에서 확인한 과거 구간의 평가',
 'intervals':u['frozen_choice_intervals']},OUT/'usage_selected_model.json')
print({'usage_MAPE':u['retrospective_2023H2_2024H1'][pick]['MAPE'],'paired_CI':u['paired_new_minus_old_MAPE_CI95'],'quarters':quarter,
 'signal_candidate_CI':s['exploratory_svr_minus_previous_CI95']})
