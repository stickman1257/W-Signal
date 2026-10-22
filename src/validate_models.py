from pathlib import Path
import argparse, json, math, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore', category=RuntimeWarning)

def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x, list): return [clean(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (float,np.floating)): return float(x) if np.isfinite(x) else None
    return x

def run(src,out):
    out.mkdir(parents=True,exist_ok=True)
    A=pd.read_csv(src/'origin/water_signals_by_legal_dong.csv')
    K=pd.read_csv(src/'origin/income_by_admin_dong.csv',dtype={'행정동코드':str})
    link=pd.read_csv(src/'public/daegu_kikmix_2024.csv',dtype=str)
    D=pd.read_csv(src/'derived/admin_signals.csv',dtype={'admin8':str})

    lk=link[link.읍면동명.notna() & link.시군구명.notna()][['시군구명','동리명','행정동코드','읍면동명']].drop_duplicates()
    lk=lk.rename(columns={'시군구명':'시군구','동리명':'읍면동'});lk['admin8']=lk.행정동코드.str[:8]
    lk=lk.merge(A[['시군구','읍면동']],on=['시군구','읍면동'])
    lk['w']=1/lk.groupby(['시군구','읍면동']).admin8.transform('nunique')
    xx=lk.merge(A,on=['시군구','읍면동']);known=xx.sig.notna()
    xx['wsig']=(xx.sig*xx.w).where(known,0);xx['den']=(xx.n_one.fillna(0)*xx.w).where(known,0)
    agg=xx.groupby('admin8')[['wsig','den']].sum();rawrate=(agg.wsig/agg.den).where(agg.den>=30)
    D['sig_rate']=D.admin8.map(rawrate)
    kfull=K[(K.거주기준==1)&(K.기준시점==202312)].set_index('행정동코드').low_rate
    D['low_rate']=D.admin8.map(kfull)
    NEWS=['복현1동','상인3동','월성2동','대명1동','송현2동','송현1동','범어2동','대명3동','신당동','대명9동']
    urban=D[D.area.eq('도시(동)') & D.W.notna()].copy()
    tests={}
    for key in ['low_rate','sig_rate','W']:
        a=urban[urban.행정동.isin(NEWS)][key];b=urban[~urban.행정동.isin(NEWS)][key]
        t=stats.mannwhitneyu(a,b,alternative='greater')
        tests[key]={'n_news':len(a),'n_other':len(b),'mean_news':a.mean(),'mean_other':b.mean(),'p_one_sided':t.pvalue}

    robustness=[]
    for w in [.25,.5,.75]:
        score=(w*urban.sig_pct+(1-w)*urban.vul_pct)*100
        ids=set(urban.assign(score=score).nlargest(20,'score').admin8)
        old=set(urban.nlargest(20,'W').admin8)
        robustness.append({'signal_weight':w,'top20_overlap':len(ids&old),'rank_rho':stats.spearmanr(score,urban.W).statistic})
    pd.DataFrame(robustness).to_csv(out/'index_sensitivity.csv',index=False,encoding='utf-8-sig')
    audit={'external_comparison':tests,'n_admin':len(D),'n_scored':int(D.W.notna().sum()),
           'n_split':int(D.split.sum()),'n_urban_split_scored':int(urban.split.sum()),
           'partial_coverage':int(((D.coverage<.9999)&D.W.notna()).sum()),
           'n_signal_only_urban':int(urban.quad.eq('② 숨은 위험').sum()),'robustness':robustness}

    pairs=link[link.읍면동명.notna() & link.시군구명.notna()][['시군구명','동리명','행정동코드']].drop_duplicates()
    pairs=pairs.rename(columns={'시군구명':'시군구','동리명':'읍면동'})
    pairs['admin8']=pairs.행정동코드.str[:8];K['admin8']=K.행정동코드.str[:8]
    def kb(r,t): return K[(K.거주기준==r)&(K.기준시점==t)].set_index('admin8').low_rate
    kcb=pd.DataFrame({'low':kb(1,202312),'low20':kb(1,202003),'low_b2':kb(2,202312)})
    legal=pairs.merge(kcb,left_on='admin8',right_index=True).groupby(['시군구','읍면동'])[['low','low20','low_b2']].mean()
    X=A.merge(legal.reset_index(),on=['시군구','읍면동'],how='left')
    X['one_rate']=X.n_one/X.n;X['log_n']=np.log10(X.n)
    X['rural']=(~X.읍면동.str.endswith(('동','가'))|X.시군구.eq('군위군')).astype(int)
    X['low_change']=X.low-X.low20
    X=X[X.rate.notna()&X.one_rate.notna()].reset_index(drop=True)
    features=['one_rate','log_n','low','low_b2','low_change','rural']
    def ridge():
        pre=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),features),
            ('gu',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['시군구'])])
        return make_pipeline(pre,RidgeCV(alphas=np.logspace(-3,3,13)))
    def cb():return CatBoostRegressor(iterations=350,depth=3,learning_rate=.03,l2_leaf_reg=8,
        cat_features=['시군구'],verbose=False,random_seed=42,thread_count=2,allow_writing_files=False)
    cvrows=[];predrows=[]
    for scheme,cv in [('무작위 5겹 5회',RepeatedKFold(n_splits=5,n_repeats=5,random_state=42)),('구군 외부검증',LeaveOneGroupOut())]:
        splits=cv.split(X,groups=X.시군구)
        ys=[]; ps={k:[] for k in ['평균값','Ridge','CatBoost','앙상블']}
        for fold,(it,iv) in enumerate(splits):
            tr,te=X.iloc[it],X.iloc[iv];y=tr.rate
            rd=ridge().fit(tr,y);pr=np.clip(rd.predict(te),0,1)

            xt=tr[features+['시군구']].copy();xv=te[features+['시군구']].copy()
            med=xt[features].median();xt[features]=xt[features].fillna(med);xv[features]=xv[features].fillna(med)
            cm=cb().fit(xt,y);pc=np.clip(cm.predict(xv),0,1)
            pred={'평균값':np.repeat(y.mean(),len(te)),'Ridge':pr,'CatBoost':pc,'앙상블':(pr+pc)/2}
            ys.extend(te.rate)
            for model,p in pred.items():
                ps[model].extend(p)
                cvrows.append({'scheme':scheme,'fold':fold,'model':model,'n_test':len(te),
                    'MAE_pp':mean_absolute_error(te.rate,p)*100})
                for i,v in zip(iv,p):predrows.append({'scheme':scheme,'fold':fold,'model':model,'row_id':int(i),'actual':X.iloc[i].rate,'prediction':v})
        for model,p in ps.items():
            cvrows.append({'scheme':scheme,'fold':'pooled','model':model,'n_test':len(ys),
                'MAE_pp':mean_absolute_error(ys,p)*100,'R2':r2_score(ys,p),'Spearman':stats.spearmanr(ys,p).statistic})
    CV=pd.DataFrame(cvrows);CV.to_csv(out/'signal_validation_predictions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(predrows).to_csv(out/'signal_oof_predictions.csv',index=False,encoding='utf-8-sig')
    audit['signal_cv']=CV[CV.fold.eq('pooled')].to_dict('records')
    audit['signal_n']=len(X)
    import joblib
    signal_ridge=ridge().fit(X,X.rate)
    joblib.dump(signal_ridge,out/'signal_ridge.joblib')
    sx=X[features+['시군구']].copy();sm=sx[features].median();sx[features]=sx[features].fillna(sm)
    cb().fit(sx,X.rate).save_model(str(out/'signal_catboost.cbm'))
    (out/'signal_features.json').write_text(json.dumps({'features':features,'cat':['시군구'],'training_medians':sm.to_dict(),
        'ensemble':'clip(0.5 * Ridge + 0.5 * CatBoost, 0, 1)','scope':'공개된 법정동 116개. 비공개 처리 지역 보완에 사용하지 않음.'},ensure_ascii=False,indent=2))

    B=pd.read_csv(src/'origin/water_usage_by_district.csv')
    B=B[B.시군구.ne('군위군')].copy();B['ym']=pd.to_datetime(B.납기년월.astype(str),format='%Y%m')
    B['grp']=B.ym.dt.month%2;B['series']=B.시군구+'_'+B.grp.astype(str)
    B=B.sort_values(['series','ym']).reset_index(drop=True)
    g=B.groupby('series').day
    for k in [1,2,3,6,7,8,9]:B[f'lag{k}']=g.shift(k)
    B['ratio']=B.lag1/B.lag7
    B['smooth_ratio']=np.median(np.array([B.lag1/B.lag7,B.lag2/B.lag8,B.lag3/B.lag9]),axis=0)
    B['month']=B.ym.dt.month;B['t']=(B.ym.dt.year-2020)*12+B.month
    for c in ['lag1','lag2','lag6']:B['log_'+c]=np.log(B[c])
    fs=['log_lag1','log_lag2','log_lag6','ratio','month','t','series']
    b=B.dropna(subset=fs+['smooth_ratio']).copy()
    def fit(train):
        m=CatBoostRegressor(iterations=400,depth=3,learning_rate=.03,l2_leaf_reg=8,cat_features=['series'],
            verbose=False,random_seed=42,thread_count=2,allow_writing_files=False)
        return m.fit(train[fs],np.log(train.day))
    def predict(m,data):
        trend=(data.lag6*data.ratio).to_numpy();cat=np.exp(m.predict(data[fs]))
        return {'직전고지':data.lag1.to_numpy(),'전년동월':data.lag6.to_numpy(),'계절추세':trend,
            '완만한계절추세':(data.lag6*data.smooth_ratio).to_numpy(),'CatBoost':cat,
            '추세75_Cat25':.75*trend+.25*cat,'추세50_Cat50':.5*trend+.5*cat}
    def metrics(y,p):return {'MAPE':float(np.mean(np.abs(y-p)/y)*100),'MAE':float(np.mean(np.abs(y-p))),
        'WAPE':float(np.sum(np.abs(y-p))/np.sum(y)*100),'bias_pct':float(np.mean((p-y)/y)*100)}
    dev=b[b.ym.between('2022-01-01','2022-12-31')]
    devpred=predict(fit(b[b.ym<'2022-01-01']),dev)
    devmet={k:metrics(dev.day.to_numpy(),p) for k,p in devpred.items()}
    selected=min(devmet,key=lambda k:devmet[k]['MAPE'])
    final=fit(b[b.ym<'2023-01-01'])
    cal=b[b.ym.between('2023-01-01','2023-06-30')]
    cp=predict(final,cal)[selected]
    err=np.abs(np.log(cal.day.to_numpy()/cp));n=len(err)
    q=float(np.sort(err)[min(n-1,math.ceil((n+1)*.90)-1)])
    q95=float(np.sort(err)[min(n-1,math.ceil((n+1)*.95)-1)])
    test=b[b.ym>='2023-07-01'];tp=predict(final,test)
    met={k:metrics(test.day.to_numpy(),p) for k,p in tp.items()}
    p=tp[selected];lo=p*np.exp(-q);hi=p*np.exp(q)
    prediction=test[['시군구','series','ym','day']].copy();prediction['prediction']=p
    prediction['lo90']=lo;prediction['hi90']=hi;prediction['covered']=(test.day>=lo)&(test.day<=hi)
    prediction['lo95']=p*np.exp(-q95);prediction['hi95']=p*np.exp(q95)
    prediction['covered95']=(test.day>=prediction.lo95)&(test.day<=prediction.hi95)
    prediction.to_csv(out/'usage_baseline_predictions.csv'.replace('予測','예측'),index=False,encoding='utf-8-sig')
    bygu=[]
    for gu,ix in test.groupby('시군구').groups.items():
        t=prediction.loc[ix];row={'gu':gu,**metrics(t.day.to_numpy(),t.prediction.to_numpy()),'n':len(t),'coverage90':t.covered.mean()}
        bygu.append(row)
    pd.DataFrame(bygu).to_csv(out/'usage_baseline_district_metrics.csv',index=False,encoding='utf-8-sig')

    fm=fit(b);hist=B[['시군구','grp','series','ym','day']].copy();future=[]
    for month in pd.date_range('2024-07-01','2024-12-01',freq='MS'):
        for series in hist[hist.grp.eq(month.month%2)].series.unique():
            h=hist[hist.series.eq(series)].sort_values('ym');v=h.day.to_numpy();r={f'lag{k}':v[-k] for k in [1,2,3,6,7,8,9]}
            r.update(ratio=r['lag1']/r['lag7'],smooth_ratio=np.median([r['lag1']/r['lag7'],r['lag2']/r['lag8'],r['lag3']/r['lag9']]),
                month=month.month,t=(month.year-2020)*12+month.month,series=series)
            for c in ['lag1','lag2','lag6']:r['log_'+c]=np.log(r[c])
            fp=float(predict(fm,pd.DataFrame([r]))[selected][0])

            future.append({'시군구':h.시군구.iloc[-1],'series':series,'ym':month.strftime('%Y-%m'),
                'prediction':fp,'bill_horizon':len(h[h.ym>'2024-06-01'])+1})
            hist=pd.concat([hist,pd.DataFrame([{'시군구':h.시군구.iloc[-1],'grp':month.month%2,'series':series,'ym':month,'day':fp}])],ignore_index=True)
    pd.DataFrame(future).to_csv(out/'usage_historical_forecast.csv',index=False,encoding='utf-8-sig')
    fm.save_model(str(out/'usage_catboost.cbm'))
    (out/'usage_baseline_model.json').write_text(json.dumps({'selected':selected,
        'formula':'lag6 * lag1 / lag7' if selected=='계절추세' else selected,
        'same_parity_billing':True,'selection_period':'2022-01~12','calibration_period':'2023-01~06',
        'calibration_q90_log':q,'calibration_q95_log':q95,
        'unit':'계량기별 일평균 사용량의 구군 중앙값 (㎥/일)'},ensure_ascii=False,indent=2))
    audit['usage']={'selected_on_2022':selected,'development':devmet,'final_test':met,'n_calibration':len(cal),'n_test':len(test),
        'calibration_q90_log':q,'coverage90':float(prediction.covered.mean()),
        'calibration_q95_log':q95,'coverage95':float(prediction.covered95.mean()),
        'mean_width90_pct':float(np.mean((hi-lo)/p)*100),'by_gu':bygu,
        'train_end':'2022-12','calibration':'2023-01~06','test':'2023-07~2024-06',
        'interpretation':'다음 동일 검침군 고지 1회 예측. 이후 실제 고지를 시차 입력으로 사용. 다회 재귀예측과 구별 설명에서 구분.'}
    (out/'model_validation.json').write_text(json.dumps(clean(audit),ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(clean(audit),ensure_ascii=False,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();run(a.source,a.output)
