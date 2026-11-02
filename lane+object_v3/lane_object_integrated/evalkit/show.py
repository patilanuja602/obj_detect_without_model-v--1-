import sys, ast, json
sys.path.insert(0, __file__.rsplit('/',2)[0])
from evalkit.evaluate_gt import evaluate, summarize
rows, D = evaluate(sys.argv[1], sys.argv[2])
for d in rows:
    print(d['frame'], d['id'].ljust(5), ('x0 %+4.0f x1 %+4.0f y1 %+4.0f y0 %s iou %s | contact err %s q=%s obs=%s' % (d['e_x0'], d['e_x1'], d['e_y1'], d['e_y0'] if d['e_y0'] is None else round(d['e_y0']), None if d['iou'] is None else round(d['iou'],2), None if d['c_err'] is None else round(d['c_err'],1), d['cq'], d['obs'])) if d['matched'] else 'MISSED')
S = summarize(rows)
print({k: (round(v['mean'],1) if isinstance(v, dict) and v else v) for k, v in S.items()})
