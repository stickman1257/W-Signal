import os
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,math,warnings,joblib
import numpy as np,pandas as pd
from scipy.special import expit,logit
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer,TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,SplineTransformer
from sklearn.linear_model import RidgeCV,Ridge,HuberRegressor
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut,GroupKFold
from sklearn.metrics import mean_absolute_error,r2_score
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore',category=RuntimeWarning)
BASE=Path(__file__).resolve().parents[1]
SRC=BASE/'data';OUT=BASE/'results';OUT.mkdir(parents=True,exist_ok=True)
def dump(obj,path):
 def clean(x):
  if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
  if isinstance(x,(list,tuple)):return [clean(v) for v in x]
  if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
  if isinstance(x,np.integer):return int(x)
  return x
 path.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False))

def signal_data():
 a=pd.read_csv(SRC/'origin/water_signals_by_legal_dong.csv');k=pd.read_csv(SRC/'origin/income_by_admin_dong.csv',dtype={'행정동코드':str})
 l=pd.read_csv(SRC/'public/daegu_kikmix_2024.csv',dtype=str)
 p=l[l.읍면동명.notna()&l.시군구명.notna()][['시군구명','동리명','행정동코드']].drop_duplicates().rename(columns={'시군구명':'시군구','동리명':'읍면동'})
 p['code']=p.행정동코드.str[:8];k['code']=k.행정동코드.str[:8]
 def kb(r,t):return k[(k.거주기준==r)&(k.기준시점==t)].set_index('code').low_rate
 kc=pd.DataFrame({'low':kb(1,202312),'low20':kb(1,202003),'low_b2':kb(2,202312)})
 leg=p.merge(kc,left_on='code',right_index=True).groupby(['시군구','읍면동'])[['low','low20','low_b2']].mean()
 x=a.merge(leg.reset_index(),on=['시군구','읍면동'],how='left');x['one_rate']=x.n_one/x.n;x['log_n']=np.log10(x.n)
 x['rural']=(~x.읍면동.str.endswith(('동','가'))|x.시군구.eq('군위군')).astype(int);x['low_change']=x.low-x.low20
 return x[x.rate.notna()&x.one_rate.notna()].reset_index(drop=True)
NUM=['one_rate','log_n','low','low_b2','low_change','rural'];CAT=['시군구']
def design(kind='linear',water=False):
 cols=['one_rate','log_n','rural'] if water else NUM
 numeric=make_pipeline(SimpleImputer(strategy='median'),StandardScaler())
 if kind=='spline':numeric=make_pipeline(SimpleImputer(strategy='median'),SplineTransformer(n_knots=3,degree=2,include_bias=False),StandardScaler())
 return ColumnTransformer([('num',numeric,cols),('gu',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT)])
def predict_signal(name,tr,te,water=False):
 cols=['one_rate','log_n','rural'] if water else NUM
 if name.startswith('blend_'):
  combos={'blend_old':['ridge','cat'],'blend_svr':['ridge','svr'],'blend_cat_svr':['cat','svr'],'blend_spline':['ridge','spline']}
  return np.mean([predict_signal(n,tr,te,water) for n in combos[name]],axis=0)
 if name in ['cat','cat_logit']:
  xt=tr[cols+CAT].copy();xv=te[cols+CAT].copy();med=xt[cols].median();xt[cols]=xt[cols].fillna(med);xv[cols]=xv[cols].fillna(med)
  m=CatBoostRegressor(iterations=350,depth=3,learning_rate=.03,l2_leaf_reg=8,cat_features=CAT,random_seed=42,verbose=False,thread_count=2,allow_writing_files=False)
  y=logit(np.clip(tr.rate,.005,.995)) if name=='cat_logit' else tr.rate
  m.fit(xt,y);p=m.predict(xv);p=expit(p) if name=='cat_logit' else p
 else:
  choices={'ridge':RidgeCV(alphas=np.logspace(-3,3,13)),
   'svr':SVR(C=.3,epsilon=.025,gamma='scale'),
   'svr_smooth':SVR(C=.1,epsilon=.02,gamma=.12),
   'spline':RidgeCV(alphas=[.1,1,10,100,1000]),
   'extra':ExtraTreesRegressor(n_estimators=180,min_samples_leaf=4,max_features=.85,random_state=42,n_jobs=2),
   'forest':RandomForestRegressor(n_estimators=180,min_samples_leaf=4,max_features=.85,random_state=42,n_jobs=2),
   'hgb':HistGradientBoostingRegressor(max_iter=160,max_leaf_nodes=5,min_samples_leaf=10,l2_regularization=8,learning_rate=.04,random_state=42)}
  m=make_pipeline(design('spline' if name=='spline' else 'linear',water),choices[name]);m.fit(tr,tr.rate);p=m.predict(te)
 return np.clip(p,0,1)

def signal():
 x=signal_data();names=['ridge','cat','cat_logit','svr','svr_smooth','spline','extra','forest','hgb','blend_old','blend_svr','blend_cat_svr','blend_spline']
 outer=LeaveOneGroupOut();pred=[];tuning=[];choices=[]
 for fold,(it,iv) in enumerate(outer.split(x,groups=x.시군구)):
  tr=x.iloc[it];te=x.iloc[iv];inner=GroupKFold(n_splits=3)
  scores={n:[] for n in names}
  for j,(a,b) in enumerate(inner.split(tr,groups=tr.시군구)):
   ti,tv=tr.iloc[a],tr.iloc[b]
   cache={n:predict_signal(n,ti,tv) for n in names if not n.startswith('blend_')}
   for n,com in {'blend_old':['ridge','cat'],'blend_svr':['ridge','svr'],'blend_cat_svr':['cat','svr'],'blend_spline':['ridge','spline']}.items():cache[n]=np.mean([cache[c] for c in com],axis=0)
   for n,p in cache.items():scores[n].extend(np.abs(tv.rate.to_numpy()-p))
  sm={n:float(np.mean(v)*100) for n,v in scores.items()};best=min(sm,key=sm.get)
  choices.append({'fold':fold,'heldout_gu':te.시군구.iloc[0],'chosen':best,'inner_MAE_pp':sm[best]})
  for n,v in sm.items():tuning.append({'fold':fold,'heldout_gu':te.시군구.iloc[0],'model':n,'inner_MAE_pp':v})
  cache={n:predict_signal(n,tr,te) for n in names if not n.startswith('blend_')}
  for n,com in {'blend_old':['ridge','cat'],'blend_svr':['ridge','svr'],'blend_cat_svr':['cat','svr'],'blend_spline':['ridge','spline']}.items():cache[n]=np.mean([cache[c] for c in com],axis=0)
  cache['nested_selection']=cache[best]
  cache['water_only_nested']=predict_signal(best,tr,te,water=True)
  for n,p in cache.items():
   for idx,v in zip(iv,p):pred.append({'fold':fold,'gu':x.iloc[idx].시군구,'legal':x.iloc[idx].읍면동,'id':int(idx),'model':n,'actual':x.iloc[idx].rate,'pred':v})
  print('signal fold',fold,te.시군구.iloc[0],best,flush=True)
 pp=pd.DataFrame(pred);pp.to_csv(OUT/'signal_predictions.csv',index=False,encoding='utf-8-sig');pd.DataFrame(tuning).to_csv(OUT/'signal_inner_selection.csv',index=False,encoding='utf-8-sig')
 metrics=[]
 for n,t in pp.groupby('model'):
  metrics.append({'model':n,'n':len(t),'MAE_pp':mean_absolute_error(t.actual,t.pred)*100,'R2':r2_score(t.actual,t.pred),'Spearman':stats.spearmanr(t.actual,t.pred).statistic})
 summary={'n':len(x),'outer':'구군별 LeaveOneGroupOut 9회','inner':'학습 구군만 GroupKFold 3겹','candidates':names,'selection_by_fold':choices,'metrics':metrics}

 rows=pp[pp.model.eq('nested_selection')].merge(pp[pp.model.eq('blend_old')][['id','pred']].rename(columns={'pred':'old'}),on='id')
 blocks=[g for _,g in rows.groupby('gu')];rng=np.random.default_rng(42);diff=[]
 for _ in range(2000):
  sample=pd.concat([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))]);diff.append((np.abs(sample.actual-sample.pred).mean()-np.abs(sample.actual-sample.old).mean())*100)
 summary['nested_minus_previous_MAE_pp_CI95']=np.quantile(diff,[.025,.975]).tolist()
 dump(summary,OUT/'signal_experiments.json');return summary

def usage_data():
 b=pd.read_csv(SRC/'origin/water_usage_by_district.csv');b=b[b.시군구.ne('군위군')].copy()
 b['ym']=pd.to_datetime(b.납기년월.astype(str),format='%Y%m');b['parity']=b.ym.dt.month%2;b['series']=b.시군구+'_'+b.parity.astype(str)
 b=b.sort_values(['series','ym']).reset_index(drop=True);g=b.groupby('series').day
 for i in range(1,10):b[f'l{i}']=g.shift(i)
 b['trend']=b.l6*b.l1/b.l7;b['smooth']=b.l6*np.median(np.array([b.l1/b.l7,b.l2/b.l8,b.l3/b.l9]),axis=0)
 b['ratio']=b.l1/b.l7;b['prev_error']=np.log(b.l1/(b.l7*b.l2/b.l8))
 b['logy']=np.log(b.day);b['resid']=np.log(b.day/b.trend);b['sin']=np.sin(2*np.pi*b.ym.dt.month/12);b['cos']=np.cos(2*np.pi*b.ym.dt.month/12)
 b['t']=(b.ym.dt.year-2020)*12+b.ym.dt.month
 for k in [1,2,3,6]:b['logl'+str(k)]=np.log(b['l'+str(k)])
 return b.dropna().copy()

UNAMES=['trend','smooth','blend50','blend75','bias_shrink','resid_ridge','resid_cat','resid_blend','direct_cat']
def usage_predict(name,tr,te):
 if name=='trend':return te.trend.to_numpy()
 if name=='smooth':return te.smooth.to_numpy()
 if name in ['blend50','blend75']:
  w=.5 if name=='blend50' else .75
  return w*te.trend.to_numpy()+(1-w)*te.smooth.to_numpy()
 if name=='bias_shrink':

  tail=tr.groupby('series').tail(6);gb=tail.groupby('series').resid.agg(['sum','count']);off=gb['sum']/(gb['count']+6)
  return te.trend.to_numpy()*np.exp(te.series.map(off).fillna(0).to_numpy())
 fs=['ratio','prev_error','sin','cos','t','series']
 if name.startswith('resid_'):
  if name=='resid_blend':return (usage_predict('resid_ridge',tr,te)+te.trend.to_numpy())/2
  if name=='resid_ridge':
   pre=ColumnTransformer([('num',make_pipeline(SimpleImputer(),StandardScaler()),fs[:-1]),('series',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['series'])])
   m=make_pipeline(pre,Ridge(alpha=20)).fit(tr[fs],tr.resid);z=m.predict(te[fs])
  else:
   m=CatBoostRegressor(iterations=220,depth=2,learning_rate=.025,l2_leaf_reg=12,loss_function='MAE',verbose=False,random_seed=42,thread_count=2,cat_features=['series'],allow_writing_files=False).fit(tr[fs],tr.resid);z=m.predict(te[fs])
  return te.trend.to_numpy()*np.exp(np.clip(z,-.2,.2))
 fs=['logl1','logl2','logl6','ratio','sin','cos','t','series']
 m=CatBoostRegressor(iterations=300,depth=3,learning_rate=.03,l2_leaf_reg=12,verbose=False,random_seed=42,thread_count=2,cat_features=['series'],allow_writing_files=False).fit(tr[fs],tr.logy)
 return np.exp(m.predict(te[fs]))
def metric(t):
 return {'n':len(t),'MAPE':float(np.mean(np.abs(t.actual-t.pred)/t.actual)*100),'MAE':float(np.mean(np.abs(t.actual-t.pred))),
  'WAPE':float(np.sum(np.abs(t.actual-t.pred))/np.sum(t.actual)*100),'bias_pct':float(np.mean((t.pred-t.actual)/t.actual)*100)}
def usage():
 b=usage_data();rows=[]

 for dt in sorted(b[b.ym>='2022-01-01'].ym.unique()):
  tr=b[b.ym<dt];te=b[b.ym.eq(dt)]
  assert tr.ym.max()<te.ym.min()
  for n in UNAMES:
   pred=usage_predict(n,tr,te)
   for (_,r),p in zip(te.iterrows(),pred):rows.append({'ym':r.ym.strftime('%Y-%m'),'gu':r.시군구,'series':r.series,'actual':r.day,'model':n,'pred':p,'train_end':tr.ym.max().strftime('%Y-%m')})
 p=pd.DataFrame(rows);sel=p[p.ym.between('2022-01','2022-12')]
 selection={n:metric(t) for n,t in sel.groupby('model')};champ=min(selection,key=lambda n:selection[n]['MAPE'])

 policy=[];policy_choices=[]
 for dt in sorted(p[p.ym>='2023-07'].ym.unique()):
  past=p[(p.ym<dt)&(p.ym>='2022-01')]
  pm={n:metric(t)['MAPE'] for n,t in past.groupby('model')};best=min(pm,key=pm.get)
  tt=p[p.ym.eq(dt)&p.model.eq(best)].copy();tt['model']='rolling_selection';policy.append(tt)
  policy_choices.append({'ym':dt,'chosen':best,'past_MAPE':pm[best]})
 p=pd.concat([p,*policy],ignore_index=True);p.to_csv(OUT/'usage_expanding_predictions.csv',index=False,encoding='utf-8-sig')
 scores={n:metric(t) for n,t in p[p.ym>='2023-07'].groupby('model')}

 cal=p[p.ym.between('2023-01','2023-06')&p.model.eq(champ)];test=p[(p.ym>='2023-07')&p.model.eq(champ)].copy()
 errs=np.abs(np.log(cal.actual/cal.pred)).to_numpy();intervals={}
 for level in [.90,.95]:
  q=float(np.sort(errs)[min(len(errs)-1,math.ceil((len(errs)+1)*level)-1)]);lo=test.pred*np.exp(-q);hi=test.pred*np.exp(q)
  intervals[str(level)]={'qlog':q,'cal_n':len(cal),'test_coverage':float(((test.actual>=lo)&(test.actual<=hi)).mean()),'width_pct':float(2*np.sinh(q)*100)}
 summary={'candidate_rules':UNAMES,'selection_2022':selection,'frozen_2022_choice':champ,'retrospective_2023H2_2024H1':scores,'rolling_policy_choices':policy_choices,'frozen_choice_intervals':intervals}
 dump(summary,OUT/'usage_experiments.json');print('usage frozen',champ,scores[champ],flush=True);return summary

if __name__=='__main__':
 print('START',flush=True);s=signal();u=usage();print('DONE',flush=True)
