from common import *
import collections,html

def run(out):
 out=Path(out);tr=readcsv(out/'q3/trials.csv');paired=readcsv(out/'q3/paired_differences.csv');buckets=collections.defaultdict(list);summary=[]
 for r in paired:buckets[tuple(r[k] for k in ['method','quantile','case_id','alpha','beta'])].append(r)
 for k,rows in buckets.items():
  for metric in ['on_time_difference','cost_difference']:
   v=np.array([float(r[metric]) for r in rows]);rng=np.random.default_rng(20261120);b=v[rng.integers(0,len(v),(2000,len(v)))].mean(1);summary.append(dict(zip(['method','quantile','case_id','alpha','beta'],k))|dict(metric=metric,mean=float(v.mean()),lo=float(np.quantile(b,.025)),hi=float(np.quantile(b,.975)),n=len(v)))
 writecsv(out/'q3/paired_summary.csv',summary)
 rank=[]
 for method in ['family_edf','family_density','ridge_edf','ridge_review_edf']:
  for q in ['0.5','0.8','0.9']:
   rr=[r for r in tr if r['method']==method and r['quantile']==q and float(r['alpha'])==1 and float(r['beta'])==.5];rank.append(dict(method=method,quantile=float(q),on_time=float(np.mean([float(r['weighted_on_time_rate']) for r in rr])),completion=float(np.mean([float(r['weighted_completion_rate']) for r in rr])),cost=float(np.mean([float(r['cost']) for r in rr]))))
 rank.sort(key=lambda x:(-x['on_time'],-x['completion'],x['cost'],x['method'],x['quantile']));dump(out/'models/frozen_selection.json',dict(selected=rank[0],ranking=rank,selection_data='validation nominal, equal case/trial mean; exploratory selection, not unbiased test performance',model_sha256=sha(out/'models/duration.json')))
 # SVG results with CSV source kept next to figure.
 curves=[]
 for r in readcsv(out/'q12/summary.csv'):
  if r.get('split')=='validation' and r['metric']=='ade' and r.get('horizon_s'):curves.append(dict(dataset=r['dataset'],model=r['model'],horizon_s=float(r['horizon_s']),ade=float(r['mean'])))
 writecsv(out/'q12/figure_source.csv',curves)
 svg=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="340"><rect width="960" height="340" fill="white"/>']
 for panel,src in enumerate(['EgoDemo','EgoPro','EgoStandard']):
  rr=[r for r in curves if r['dataset']==src];mx=max(r['ade'] for r in rr)*1.15 or 1;x0=panel*320+40;svg.append(f'<text x="{x0}" y="24">{src}: validation ADE (source units)</text><path d="M{x0},40 V290 H{x0+250}" fill="none" stroke="black"/>')
  for method,color in [('hold','#2563eb'),('constant_velocity','#dc2626')]:
   pts=sorted([r for r in rr if r['model']==method],key=lambda r:r['horizon_s']);poly=' '.join(f'{x0+r["horizon_s"]*230},{290-r["ade"]/mx*230}' for r in pts);svg.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2"/>')
   for r in pts:svg.append(f'<text font-size="10" x="{x0+r["horizon_s"]*230}" y="{285-r["ade"]/mx*230}">{r["ade"]:.3g}</text>')
  svg.append(f'<text x="{x0}" y="320" fill="#2563eb">hold</text><text x="{x0+60}" y="320" fill="#dc2626">constant velocity</text>')
 svg.append('</svg>');(out/'q12/validation_ade.svg').write_text(''.join(svg),encoding='utf8')
 print('paired comparisons, frozen selection and figures complete',flush=True)
