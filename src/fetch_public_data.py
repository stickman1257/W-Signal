from pathlib import Path
from urllib.request import urlopen
from datetime import datetime,timezone,timedelta
import argparse,hashlib,html,json,re,io
import pandas as pd
from scipy.stats import spearmanr

WELFARE='https://www.daegu.go.kr/welf/index.do?menu_id=00000702'
DONGGU='https://www.data.go.kr/data/15134142/fileData.do'

def get(url):
    with urlopen(url,timeout=40) as r:return r.read()

def run(src,out):
    out.mkdir(parents=True,exist_ok=True)
    asof=datetime.now(timezone(timedelta(hours=9))).date().isoformat()
    wb=get(WELFARE);s=wb.decode('utf-8')
    table=next(t for t in re.findall(r'<table\b[\s\S]*?</table>',s) if '종합사회복지관 현황' in t)
    rows=[];gu=''
    for tr in re.findall(r'<tr\b[\s\S]*?</tr>',table):
        cells=[re.sub(r'\s+',' ',html.unescape(re.sub('<[^>]*>',' ',x))).strip()
            for x in re.findall(r'<td\b[^>]*>([\s\S]*?)</td>',tr)]
        if not cells:continue
        if len(cells)==11:gu=re.sub(r'\(.*','',cells.pop(0))
        if len(cells)!=10:raise ValueError('공식 시설 표의 구조가 바뀌었습니다. 주소·전화 열을 확인해주세요.')
        rows.append({'gu':gu,'name':cells[0]+'종합사회복지관','address':'대구광역시 '+cells[-2],
            'phone':'053-'+cells[-1],'asof':asof+' 열람','source':WELFARE})
    pd.DataFrame(rows).to_csv(out/'welfare_centers.csv',index=False,encoding='utf-8-sig')
    page=get(DONGGU).decode('utf-8')
    link=re.search(r'"contentUrl"\s*:\s*"([^"]+)"',page)
    if not link:raise ValueError('공식 다운로드 링크를 확인해주세요.')
    url=html.unescape(link.group(1));raw=get(url)
    try:pub=pd.read_csv(io.BytesIO(raw),encoding='utf-8-sig')
    except UnicodeDecodeError:pub=pd.read_csv(io.BytesIO(raw),encoding='cp949')
    pub.to_csv(out/'donggu_elderly_unclaimed_deaths.csv',index=False,encoding='utf-8-sig')

    pub=pub.copy();pub['행정동']=pub.행정동.replace({'불로봉무동':'불로.봉무동'})
    d=pd.read_csv(src/'derived/admin_signals.csv',dtype={'admin8':str})
    m=d[d.시군구.eq('동구')].merge(pub[pub.연도.eq(2023)],on='행정동',how='left',validate='1:1')
    future=pub[pub.연도.eq(2025)][['행정동','무연고사망자수(사망장소 기준)']].rename(
        columns={'무연고사망자수(사망장소 기준)':'2025_무연고사망_장소기준'})
    m=m.merge(future,on='행정동',how='left',validate='1:1')
    m.to_csv(out/'donggu_linked_data.csv',index=False,encoding='utf-8-sig')
    results={}
    for col in ['sig_rate','low_rate','W']:
        a=m.dropna(subset=[col,'2025_무연고사망_장소기준'])
        v=spearmanr(a[col],a['2025_무연고사망_장소기준'])
        results[col]={'n':len(a),'rho':float(v.statistic),'p_two_sided':float(v.pvalue)}
    meta={'retrieved':asof,'welfare':{'url':WELFARE,'rows':len(rows),'sha256':hashlib.sha256(wb).hexdigest()},
        'donggu':{'url':DONGGU,'download':url,'rows':len(pub),'sha256':hashlib.sha256(raw).hexdigest()},
        'match_2023_elderly':int(m.노인인구수.notna().sum()),'exploratory_comparison_2025':results,
        'scope':'사망장소 기준 무연고 사망 통계는 고독사 또는 노인 사망률과 정의가 다르며 탐색 대조에 사용.'}
    (out/'public_data_linkage.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print(json.dumps(meta,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
