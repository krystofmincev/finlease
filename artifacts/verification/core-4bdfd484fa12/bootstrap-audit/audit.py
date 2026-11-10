"""Isolated, offline conditional-bootstrap feasibility and member-precision spike."""
import time
START = time.perf_counter()
import hashlib
import importlib.metadata
import json
import resource
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom
from sklearn.ensemble import HistGradientBoostingClassifier

from lease_lab.artifacts.store import FrozenBundle, digest, encode, source_digest, write_json
from lease_lab.builds.exact import support_profile
from lease_lab.config import CONFIG, HISTORICAL_OFFERS, ROOT
from lease_lab.contracts import Context, Finance
from lease_lab.data.events import Event
from lease_lab.data.features import Transform
from lease_lab.decisions.support import support_for
from lease_lab.evaluation.cohorts import fit_subsets, offline_rows
from lease_lab.evaluation.point_models import PointDesign
from lease_lab.models.beta_cells import fit_beta
from lease_lab.report_contracts import SelectionCheckpoint, selection_payload

HERE = Path(__file__).resolve().parent
LIMIT = 600
MARGINS = np.array([CONFIG['acceptance_contribution_cents'][o] for o in HISTORICAL_OFFERS])
PROBES = [Context(), Context(segment='new', business_age_years=1.5, monthly_turnover_cents=1800000, existing_debt_burden=.25),
          Context(segment='restart', business_age_years=5., monthly_turnover_cents=2500000, existing_debt_burden=.18)]
RESULT = {'status':'started', 'scientific_fit_count':0, 'fixture_fit_count':0, 'families':{},
          'coverage_status':'not_opened', 'fresh_final_status':'not_opened', 'advisory_enabled':False}


class StopAudit(RuntimeError):
    pass


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def guard():
    if time.perf_counter()-START >= LIMIT:
        raise StopAudit('wall_budget_cutoff')
    if rss() > 1_500_000_000:
        raise StopAudit('rss_stop')


def save():
    RESULT.update(elapsed_seconds=time.perf_counter()-START, peak_rss_bytes=rss())
    write_json(HERE/'RESULT.json', RESULT)


def grouped_indices(groups, draws):
    return np.concatenate([groups[i] for i in draws])


def paired_delta(p):
    utilities = p*MARGINS - CONFIG['decision_cost_cents']
    return utilities - utilities[..., 1, None]


def queries(design, contexts, offers):
    rows = [Event(applicant_id=f'probe-{i}',split='test',context=c,offer_id=o,
                  assignment_probability=None,decision_at=0,outcome_at=0,available_at=0,accepted=None)
            for i,(c,o) in enumerate(zip(contexts,offers))]
    return design.matrix(rows)


def predict(model, matrix):
    p = model.predict_proba(matrix)[:, int(np.flatnonzero(model.classes_ == 1)[0])]
    if p.shape != (len(matrix),) or not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
        raise StopAudit('invalid_member_predictions')
    return p


class Fitter:
    def __init__(self, rows, design, parameters):
        if any(e.split != 'fit' or e.accepted is None or e.available_at > 6000 for e in rows):
            raise ValueError('fit-only mature factual responses required')
        self.X = design.matrix(rows)
        self.y = np.array([e.accepted for e in rows],dtype=int)
        if set(self.y) != {0,1}:
            raise ValueError('binary fit labels required')
        by_id = {}
        for i,e in enumerate(rows): by_id.setdefault(e.applicant_id,[]).append(i)
        self.groups = [np.array(indices) for indices in by_id.values()]
        self.parameters = {**parameters,'early_stopping':False,'warm_start':False}

    def fit(self, indices):
        guard()
        if set(self.y[indices]) != {0,1}:
            raise StopAudit('single_class_member; no replacement/retry')
        before = time.perf_counter()
        model = HistGradientBoostingClassifier(**self.parameters).fit(self.X[indices],self.y[indices])
        RESULT['scientific_fit_count'] += 1
        elapsed = time.perf_counter()-before
        if model.n_iter_ != self.parameters['max_iter']:
            raise StopAudit('fixed_round_count_mismatch')
        guard()
        return model,elapsed


def rank_interval(values, confidence=.95):
    n=len(values);alpha=1-confidence
    low=int(binom.ppf(alpha/2,n,.05));high=int(binom.ppf(1-alpha/2,n,.05))+1
    ordered=np.sort(values,axis=0)
    return {'lower_rank':low,'upper_rank':high,
            'lower':ordered[low-1].tolist() if low>=1 else None,
            'upper':ordered[high-1].tolist() if high<=n else None}


def main():
    assert ROOT == HERE/'snapshot'
    assert source_digest() == '1ec33f9b4156ba16945c6221864c66635dbf4c134a882e7b183f69cdeed7c9c7'
    checkpoint=SelectionCheckpoint.model_validate_json((HERE/'point-selection.json').read_bytes())
    assert digest(encode(selection_payload(checkpoint))) == checkpoint.comparison.freeze_sha256
    baseline=FrozenBundle.model_validate_json((HERE/'baseline-bundle.json').read_bytes())
    np.testing.assert_array_equal(grouped_indices([np.array([0,1]),np.array([2])],[0,0,1]),[0,1,0,1,2])
    np.testing.assert_array_equal(paired_delta(np.array([[.8,.5,.6],[.2,.5,.1]])),[[9500.,0.,39500.],[-44500.,0.,-45500.]])
    assert not np.array_equal(paired_delta(np.array([[.8,.5,.1],[.2,.5,.6]])),[[9500.,0.,39500.],[-44500.,0.,-45500.]])
    assert rank_interval(np.zeros((512,1)))['lower_rank'] == 16 and rank_interval(np.zeros((512,1)))['upper_rank'] == 37
    assert rank_interval(np.zeros((512,1)),.975)['lower_rank'] == 15 and rank_interval(np.zeros((512,1)),.975)['upper_rank'] == 38
    commitment={'method':'whole-applicant-percentile-bootstrap-fixed-recipe-v1','baseline_run':checkpoint.provenance.run_id,
        'source_sha256':source_digest(),'point_freeze_sha256':checkpoint.comparison.freeze_sha256,
        'proposal_sha256':(HERE/'proposal.sha256').read_text().strip(),'script_sha256':digest(Path(__file__).read_bytes()),
        'scenario_versions':checkpoint.comparison.scenario_versions,'member_root':810000,'member_coordinates':['root','family_index','stream_index'],
        'family_order':['smooth','nonlinear'],'streams':[0,1],'member_counts':[32,128,256,512],
        'probes':[c.model_dump(mode='json') for c in PROBES],'offers':HISTORICAL_OFFERS,'quantiles':'linear',
        'probability_q05_tolerance':.02,'paired_delta_q05_tolerance_cents':200.,'wall_seconds':600,
        'rss_target_bytes':1073741824,'rss_stop_bytes':1500000000,'max_scientific_fits':33856,
        'coverage_recipe':{'seeds':list(range(820000,820030)),'namespace':'gbdt-bootstrap-coverage-v1-{family}-{replicate}'},
        'conditional_final_recipe':{'seed':20261112,'namespace':'gbdt-bootstrap-confirmation-v1'},'advisory_enabled':False}
    write_json(HERE/'METHOD-COMMITMENT.json',commitment)
    RESULT.update(commitment_sha256=digest(encode(commitment)),packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn']},
                  interpretation='Conditional resampling variability, not posterior confidence; selected recipe/external transform fixed. No Bernoulli outcome noise added.')
    state={}
    for fi,family in enumerate(['smooth','nonlinear']):
        before=time.perf_counter();rows=offline_rows(family);fit=[e for e in rows if e.split=='fit'];fit=fit_subsets(fit)[3600]
        m=next(m for m in checkpoint.comparison.methods if m.family==family and m.model=='gbdt-point-selected')
        assert digest(encode([e.model_dump(mode='json') for e in fit])) == m.fit_data_sha256
        design=PointDesign(Transform(mean=m.feature.base_mean,scale=m.feature.base_scale),m.feature)
        fitter=Fitter(fit,design,m.parameters);assert digest(fitter.X.tobytes()) == m.fit_matrix_sha256
        generation_seconds=time.perf_counter()-before
        profile=support_profile(fit);beta=fit_beta(fit)
        supported=np.array([[support_for(profile,beta,c,Finance(),o,'nov-2026',True).automated for o in HISTORICAL_OFFERS] for c in PROBES])
        probe_X=queries(design,[c for c in PROBES for _ in HISTORICAL_OFFERS],list(HISTORICAL_OFFERS)*3)
        factual_X=design.matrix(fit[:900])
        counter_X=queries(design,[e.context for e in fit[:900] for _ in HISTORICAL_OFFERS],list(HISTORICAL_OFFERS)*900)
        before=time.perf_counter();original,original_fit_seconds=fitter.fit(np.arange(len(fit)));original_p=predict(original,probe_X).reshape(3,3);del original
        rngs=[np.random.default_rng(np.random.SeedSequence([810000,fi,s])) for s in [0,1]]
        values=[[],[]];hashes=[[],[]];times=[]
        for _ in range(32):
            draws=rngs[0].integers(0,len(fitter.groups),len(fitter.groups));indices=grouped_indices(fitter.groups,draws)
            model,fit_s=fitter.fit(indices)
            before=time.perf_counter();p=predict(model,probe_X).reshape(3,3);probe_s=time.perf_counter()-before
            before=time.perf_counter();predict(model,factual_X);coverage_s=time.perf_counter()-before
            before=time.perf_counter();predict(model,counter_X);counter_s=time.perf_counter()-before
            values[0].append(p);hashes[0].append(digest(indices.tobytes()));times.append([fit_s,probe_s,coverage_s,counter_s]);del model
        means=np.mean(times,axis=0)
        expected=(1024*(means[0]+means[1])+original_fit_seconds + 30*(512*(means[0]+means[1]+means[2])+original_fit_seconds+generation_seconds)
                  +512*(means[0]+means[1]+means[2]+means[3])+original_fit_seconds)
        RESULT['families'][family]={'original_point':original_p.tolist(),'fit_groups':len(fitter.groups),'fit_rows':len(fit),
            'support':supported.tolist(),'feature_sha256':m.feature.transform_sha256,'fit_data_sha256':m.fit_data_sha256,
            'selected_parameters':m.parameters,'cost_probe_members':32,'mean_seconds':dict(zip(['fit','probe_prediction','coverage_factual_prediction','final_counteroffer_prediction'],means.tolist())),
            'generation_seconds':generation_seconds,'projected_full_family_seconds':float(expected),'actual_members':[32,0]}
        state[family]=(fitter,probe_X,rngs,values,hashes,supported)
        save()
    projection=sum(v['projected_full_family_seconds'] for v in RESULT['families'].values())+5.
    RESULT['projected_full_seconds']=projection
    if projection > LIMIT:
        RESULT['status']='cost_cap_infeasible';save();return
    for family,(fitter,X,rngs,values,hashes,supported) in state.items():
        for stream in [0,1]:
            while len(values[stream]) < 512:
                indices=grouped_indices(fitter.groups,rngs[stream].integers(0,len(fitter.groups),len(fitter.groups)))
                model,_=fitter.fit(indices);values[stream].append(predict(model,X).reshape(3,3));hashes[stream].append(digest(indices.tobytes()));del model
                RESULT['families'][family]['actual_members'][stream]=len(values[stream])
        assert len(set(hashes[0])) == 512 and len(set(hashes[1])) == 512
        arrays=[np.asarray(v) for v in values]
        records=[]
        for n in [32,128,256,512]:
            p05=[np.quantile(v[:n],.05,axis=0,method='linear') for v in arrays]
            delta=[paired_delta(v[:n]) for v in arrays]
            d05=[np.quantile(v,.05,axis=0,method='linear') for v in delta]
            pdiff=np.abs(p05[0]-p05[1]);ddiff=np.abs(d05[0]-d05[1]);mask=supported.copy();mask[:,1]=False
            records.append({'B':n,'probability_q05_streams':[p.tolist() for p in p05],'paired_delta_q05_streams_cents':[d.tolist() for d in d05],
                'max_supported_probability_difference':float(pdiff[supported].max()),'max_supported_delta_difference_cents':float(ddiff[mask].max()),
                'probability_rank_interval_stream0':rank_interval(arrays[0][:n]),'paired_delta_joint_rank_interval_stream0':rank_interval(delta[0],.975)})
        RESULT['families'][family]['precision']=records
        RESULT['families'][family]['member_hashes']=hashes
        for stream,array in enumerate(arrays):np.save(HERE/f'{family}-stream{stream}.npy',array,allow_pickle=False)
        save()
    bad=any(f['precision'][-1]['max_supported_probability_difference'] > .02 or f['precision'][-1]['max_supported_delta_difference_cents'] > 200 for f in RESULT['families'].values())
    RESULT['status']='tail_unresolved' if bad else 'precision_passed_requires_independent_validation'
    save()


try:
    main()
except StopAudit as error:
    RESULT['status']=str(error);save()
except Exception as error:
    RESULT['status']='technical_failure';RESULT['error']=f'{type(error).__name__}: {error}';save();raise
print(json.dumps({k:v for k,v in RESULT.items() if k!='families'},indent=2),flush=True)
