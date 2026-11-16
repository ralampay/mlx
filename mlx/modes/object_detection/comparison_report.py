"""Complete-pair inference and descriptive tables for architecture suites."""
import html
import itertools
import math
from pathlib import Path
from statistics import mean, stdev

from mlx.core.artifacts import write_csv, write_json_atomic
from mlx.core.paired_statistics import AnalyzePairedDifferences
from mlx.core.paired_superiority import holm_adjust
from .comparison_suite import read_json


class AnalyzeComparisonSuite:
    def __init__(self, output):
        self.root = Path(output)

    def execute(self):
        protocol = read_json(self.root/'protocol.json')
        rows, claims = [], []
        for dataset in protocol['datasets']:
            for seed in protocol['seeds']:
                for model in protocol['models']:
                    path = self.root/'runs'/dataset/model/f'seed-{seed}'/'complete.json'
                    if path.exists():
                        record = read_json(path)
                        rows.append({k:v for k,v in record.items() if k != 'metrics'} | record['metrics'])
        expected = len(protocol['datasets'])*len(protocol['models'])*len(protocol['seeds'])
        complete = len(rows) == expected
        if complete:
            for dataset in protocol['datasets']:
                for control, candidate in itertools.combinations(protocol['models'], 2):
                    differences = []
                    for seed in protocol['seeds']:
                        a = next(r for r in rows if r['dataset']==dataset and r['seed']==seed and r['model']==candidate)
                        b = next(r for r in rows if r['dataset']==dataset and r['seed']==seed and r['model']==control)
                        differences.append(100*(a['map_50_95']-b['map_50_95']))
                    if stdev(differences) == 0:
                        statistics = dict(mean_difference=mean(differences), note='Degenerate variance; inferential claims withheld.')
                        p_difference = p_equivalence = 1.0
                    else:
                        statistics = AnalyzePairedDifferences(differences, equivalence_margin=protocol['equivalence_margin_pp']).execute().to_dict()
                        p_difference = statistics['exact_sign_flip_p_two_sided']
                        p_equivalence = max(statistics['tost_lower_p'],statistics['tost_upper_p'])
                    for kind, p in (('superiority',p_difference),('equivalence',p_equivalence)):
                        claims.append(dict(dataset=dataset,candidate=candidate,control=control,kind=kind,
                                           p=p if math.isfinite(p) else 1.0, statistics=statistics))
            for claim, adjusted in zip(claims,holm_adjust([c['p'] for c in claims])):
                claim['holm_p'] = adjusted
                claim['supported'] = adjusted < protocol['alpha']
                if claim['kind']=='superiority':
                    claim['favored_model'] = claim['candidate'] if claim['statistics']['mean_difference']>0 else claim['control']
        result = dict(completed=len(rows), expected=expected, status='completed' if complete else 'incomplete',claims=claims)
        output = self.root/'aggregate'
        output.mkdir(exist_ok=True)
        write_json_atomic(output/'results.json',rows)
        if rows:
            fields = sorted({k for r in rows for k in r})
            write_csv(output/'results.csv',rows,fieldnames=fields)
        write_json_atomic(output/'statistics.json',result)
        lines = ['# Scratch architecture comparison','',f'Completed: {len(rows)}/{expected}.',
                 '', 'Fixed-split, 100-epoch-budget interpretation; not a guarantee of convergence or unseen-dataset generalization.',
                 'No inferential conclusions until every planned seed triple is complete.', '',
                 '| Dataset | Model | Seeds | AP50–95 mean ± SD | AP50 | Train minutes |',
                 '|---|---|---:|---:|---:|---:|']
        for dataset in protocol['datasets']:
            for model in protocol['models']:
                group = [r for r in rows if r['dataset']==dataset and r['model']==model]
                if group:
                    ap = [100*r['map_50_95'] for r in group]
                    lines.append(f"| {dataset} | {model} | {len(group)} | {mean(ap):.2f} ± {stdev(ap) if len(ap)>1 else 0:.2f} | {100*mean(r['map_50'] for r in group):.2f} | {mean(r['training_seconds'] for r in group)/60:.1f} |")
        for claim in claims:
            lines.append(f"\n- {claim['dataset']}: {claim['candidate']} vs {claim['control']}, {claim['kind']}: difference {claim['statistics']['mean_difference']:+.3f} AP points; Holm p={claim['holm_p']:.6g}; supported={claim['supported']}.")
        content = '\n'.join(lines)+'\n'
        (output/'summary.md').write_text(content)
        (output/'summary.html').write_text('<!doctype html><meta charset="utf-8"><title>Architecture study</title><pre>'+html.escape(content)+'</pre>')
        return result
