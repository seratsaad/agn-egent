import sys, os; sys.path.insert(0,'.')
import config; config.pin_threads()
from agn_egent import (fetch_sdss_spectrum, query_shen, run_agent, RuleInspector,
                       RuleContinuumPlanner, render_diagnostic, derive)
OUT='data/runs/gallery'; os.makedirs(OUT, exist_ok=True)
# AGN-dominated, decent-SNR quasars spanning a range of z
OBJS=[(650,52143,166),(388,51793,445),(651,52141,535),(2630,54327,149)]
for p,m,f in OBJS:
    oid=f'{p}-{m}-{f}'
    try:
        shen=query_shen(p,m,f); spec=fetch_sdss_spectrum(p,m,f,name=oid)
        out=run_agent(spec, inspector=RuleInspector(),
                      continuum_planner=RuleContinuumPlanner(),
                      workdir=os.path.join(OUT,oid))
        r=out.final_result
        render_diagnostic(r, os.path.join(OUT,oid+'.png'), report=out.final_verdict)
        d=derive(r,'Hb')
        nrem=sum(1 for s in out.steps if s.decision.remedy and s.decision.remedy.get('action')=='remove_line')
        print('%s z=%.3f | logL5100=%.2f (Shen %.2f) | logMBH=%.2f (Shen %.2f) | iters=%d noise-lines-removed=%d'
              % (oid, spec.z, r.continuum.get('LogL5100',float('nan')), shen.log_L5100,
                 d.log_MBH if d else float('nan'), shen.log_MBH_hbeta, out.n_iterations, nrem))
    except Exception as e:
        print('  [skip]', oid, e)
print('FIGS in', OUT)
