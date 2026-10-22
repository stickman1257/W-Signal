import argparse
import csv
import json
import math
import random
from pathlib import Path


def build(root):
    rng = random.Random(2026)
    source = json.loads((root / 'data/public/daegu_adm2024.geojson').read_text())
    geo, dongs = [], []
    features = sorted(source['features'], key=lambda x: x['properties']['adm_cd2'])
    missing = set(rng.sample(range(len(features)), 15))
    for i, feature in enumerate(features):
        props, geometry = feature['properties'], feature['geometry']
        polygons = geometry['coordinates'] if geometry['type'] == 'MultiPolygon' else [geometry['coordinates']]
        id_ = props['adm_cd2'][:8]
        name, gu = props['adm_nm'].split()[-1], props['sggnm']
        rings = [[[round(x, 5), round(y, 5)] for x, y in polygon[0]] for polygon in polygons]
        geo.append({'id': id_, 'name': name, 'gu': gu, 'rings': rings})
        dongs.append({'id': id_, 'name': name, 'gu': gu, 'area': '도시(동)' if name.endswith('동') else '농촌(읍·면)',
                      'sig': None if i in missing else round(rng.uniform(.08, .40), 4),
                      'low': round(rng.uniform(.18, .50), 4), 'one': round(rng.uniform(.15, .55), 4),
                      'W': None, 'q': 0, 'sp': None, 'lp': None,
                      'split': rng.random() < .6, 'coverage': 1.0 if rng.random() < .8 else round(rng.uniform(.55, .95), 4)})
    for area in ['도시(동)', '농촌(읍·면)']:
        group = [x for x in dongs if x['area'] == area and x['sig'] is not None]
        sig_order, low_order = sorted(x['sig'] for x in group), sorted(x['low'] for x in group)
        sig_median = (sig_order[(len(group)-1)//2] + sig_order[len(group)//2])/2
        low_median = (low_order[(len(group)-1)//2] + low_order[len(group)//2])/2
        for d in group:
            d['sp'] = (sig_order.index(d['sig']) + 1)/len(group)
            d['lp'] = (low_order.index(d['low']) + 1)/len(group)
            d['W'] = round(100*d['sp']*d['lp'], 1)
            d['q'] = 1 if d['sig'] >= sig_median and d['low'] >= low_median else 2 if d['sig'] >= sig_median else 3 if d['low'] >= low_median else 4
    gus = sorted(set(d['gu'] for d in dongs))
    history, tests, future = [], [], []
    for gi, gu in enumerate(gus):
        base = .35 + gi*.13
        for t in range(54):
            year, month = 2020 + t//12, t%12+1
            if gu == '군위군' and year < 2024:
                continue
            actual = round(base*(1 + .08*math.sin(2*math.pi*month/12))*(1+.002*t)*rng.uniform(.985,1.015), 4)
            history.append({'gu': gu, 'ym': f'{year}{month:02}', 'parity': month%2, 'day': actual})
            if 202307 <= year*100+month <= 202406 and gu != '군위군':
                pred = round(actual*rng.uniform(.975,1.025), 5)
                tests.append({'gu': gu, 'ym': f'{year}{month:02}', 'parity': month%2, 'pred': pred,
                              'lo90': pred*.94, 'hi90': pred*1.06, 'lo95': pred*.90, 'hi95': pred*1.10})
        if gu != '군위군':
            for month in range(7,13):
                pred = round(base*(1+.08*math.sin(2*math.pi*month/12))*1.11,5)
                future.append({'gu': gu, 'ym': f'2024{month:02}', 'parity': month%2, 'pred': pred, 'horizon': (month-5)//2})
    resources = list(csv.DictReader((root/'results/welfare_centers.csv').open(encoding='utf-8-sig')))
    summary = json.loads((root/'results/model_summary.json').read_text())
    audit = {'signal_cv': summary['signal_cv'], 'signal_n': summary['signal_n'], 'usage': summary['usage'],
             'n_admin': len(dongs), 'n_scored': sum(d['W'] is not None for d in dongs),
             'n_split': sum(d['split'] for d in dongs), 'n_urban_split_scored': sum(d['split'] and d['W'] is not None for d in dongs),
             'partial_coverage': sum(d['coverage'] < 1 and d['W'] is not None for d in dongs),
             'n_signal_only_urban': sum(d['q'] == 2 and d['area'] == '도시(동)' for d in dongs), 'robustness': []}
    data = {'demo': True, 'demo_seed': 2026, 'dongs': dongs, 'geo': geo, 'history': history,
            'mainParity': {gu: 0 for gu in gus}, 'usageTest': tests, 'future': future,
            'resources': resources, 'elderly': {}, 'audit': audit}
    (root/'app/src/data.js').write_text('const DATA = '+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n')
    html = (root/'app/src/index.html').read_text()
    html = html.replace('<link rel="stylesheet" href="styles.css">', '<style>'+(root/'app/src/styles.css').read_text()+'</style>')
    html = html.replace('<script src="data.js" defer></script>', '<script>'+(root/'app/src/data.js').read_text()+'</script>')
    html = html.replace('<script src="app.js" defer></script>', '<script>'+(root/'app/src/app.js').read_text()+'</script>')
    (root/'app/index.html').write_text(html)
    print(f'Synthetic demo: {len(dongs)} regions, {len(history)} observations, seed=2026')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    build(parser.parse_args().root)


if __name__ == '__main__':
    main()
