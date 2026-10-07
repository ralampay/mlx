"""Fixed-split overall-AP claims, separate from training and descriptive reports."""
import html
import json
import math
from pathlib import Path
from statistics import mean, stdev

from mlx.core.artifacts import write_json_atomic
from mlx.core.exceptions import MLXUserError
from mlx.core.paired_statistics import AnalyzePairedDifferences
from mlx.core.paired_superiority import holm_adjust
from mlx.modes.object_detection.adapter_report import GenerateAdapterReport


class GeneratePairedDawnReport:
    def __init__(self, config, *, final=False):
        self.config, self.final = config, final

    def execute(self):
        from scipy import stats
        c = self.config
        root = Path(c['output'])
        runs, groups = [], {}
        for condition in c['conditions']:
            group = {}
            for seed in c['seeds']:
                directory = root/'runs'/condition['id']/condition['method']/f'seed-{seed}'
                path = directory/'metrics.json'
                if not path.exists():
                    continue
                row = json.loads(path.read_text())
                if row.get('status') != 'completed':
                    continue
                if row['seed'] != seed or row['method'] != condition['method']:
                    raise MLXUserError(f'Run identity mismatch: {path}')
                group[seed] = float(row['mAP50_95'])
                runs.append({**row,'method':condition['id'],'source_run':str(directory)})
            groups[condition['id']] = group
        if not runs:
            raise MLXUserError('No completed DAWN runs to report.')
        if self.final and any(set(group) != set(c['seeds']) for group in groups.values()):
            raise MLXUserError('Final inference requires every prespecified seed for all four methods.')
        GenerateAdapterReport(root, comparison_method='drax-residual-fusion', extra_runs=tuple(runs)).execute()
        summaries = {}
        for method, group in groups.items():
            if not group:
                continue
            values = list(group.values())
            average = mean(values)
            sd = stdev(values) if len(values)>1 else None
            width = float(stats.t.ppf(.975,len(values)-1))*sd/math.sqrt(len(values)) if sd is not None else None
            summaries[method] = {'n':len(values),'mean':average,'sd':sd,
                                 'ci95':[average-width,average+width] if width is not None else None}
        claims = []
        if self.final:
            fusion = groups['drax-residual-fusion']
            for comparator, group in groups.items():
                if comparator == 'drax-residual-fusion':
                    continue
                differences = [fusion[s]-group[s] for s in c['seeds']]
                result = AnalyzePairedDifferences(differences, equivalence_margin=c['analysis']['equivalence_margin'],
                                                  bootstrap_draws=10000).execute().to_dict()
                if result['standard_deviation'] == 0:
                    raise MLXUserError('Zero paired variance: inspect results before statistical claims.')
                superiority_p = float(stats.ttest_1samp(differences, 0, alternative='greater').pvalue)
                claims.extend([
                    {'comparator':comparator,'claim':'superiority','p_raw':superiority_p,
                     'mean_difference':result['mean_difference'],'details':result},
                    {'comparator':comparator,'claim':'equivalence','p_raw':max(result['tost_lower_p'],result['tost_upper_p']),
                     'mean_difference':result['mean_difference'],'details':result}])
            for claim, adjusted in zip(claims,holm_adjust(row['p_raw'] for row in claims)):
                claim['p_holm'] = adjusted
                claim['supported'] = adjusted < c['analysis']['family_alpha']
        result = {'final':self.final,'analysis':c['analysis'],'method_summaries':summaries,'claims':claims,
                  'scope':'Training-seed variation on one fixed, previously examined DAWN split; not new-image uncertainty.',
                  'historical_runs_used':False,'run_count':len(runs)}
        aggregate = root/'aggregate'
        write_json_atomic(aggregate/'statistical-analysis.json',result)
        lines = ['# DAWN overall-AP paired comparison','',
                 f"Completed training runs: {len(runs)}/80. Final inference: {self.final}.",'',
                 '20 fresh paired seeds; 20 epochs; best validation checkpoint; overall test AP50–95.',
                 'Equivalence margin: ±1 AP point. Six prespecified claims (three superiority, three equivalence), Holm family-wise α=0.05. AP50/precision/recall/resource metrics are descriptive only.',
                 'No optional stopping and no small-object claims. Existing test split has been examined before; fresh seeds do not make it a new independent test dataset.','',
                 '| Method | Seeds | AP50–95 mean ± SD (points) |','|---|---:|---:|']
        for method, values in summaries.items():
            deviation = f"{values['sd']*100:.2f}" if values['sd'] is not None else 'NA'
            lines.append(f"| {method} | {values['n']} | {values['mean']*100:.2f} ± {deviation} |")
        lines += ['','## Final statistical claims','']
        if not self.final:
            lines.append('Withheld until all 80 training runs complete. Interim means are descriptive, not stopping rules.')
        for claim in claims:
            lines.append(f"- Fusion vs {claim['comparator']}: {claim['claim']}; difference {claim['mean_difference']*100:+.2f} AP points; Holm p={claim['p_holm']:.6g}; supported={claim['supported']}.")
        lines += ['', 'Failure to establish superiority does not establish equivalence. Both claims can hold for a small positive effect. Use adjusted claim flags, not unadjusted decisions in diagnostic details. The sample size is fixed by budget, not a guarantee of adequate power.']
        report = '\n'.join(lines)+'\n'
        (aggregate/'statistical-report.md').write_text(report)
        (aggregate/'statistical-report.html').write_text('<!doctype html><meta charset="utf-8"><title>DAWN paired study</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto}pre{white-space:pre-wrap}</style><pre>'+html.escape(report)+'</pre>')
        return result
