import { useState } from 'react';
import { downloadEvidence } from '../api/client';
import type { Evaluation } from '../api/client';
import { euro, percent } from '../components/format';
import './audit.css';
import { PointDevelopment, comparisonModels } from './PointDevelopment';
import type { ComparisonView } from './PointDevelopment';

type Family = 'smooth' | 'nonlinear';
type Prediction = Evaluation['prediction'][number];
type Coverage = Evaluation['coverage'][number];
type Finding = Evaluation['findings'][number];
type CalibrationBin = Prediction['calibration'][number];

const modelLabels: Record<string, string> = {
  'empirical-rate': 'Observed by rate',
  'empirical-cell': 'Observed by cell',
  beta: 'Exact Beta cells',
  'logistic-point': 'Logistic point estimate',
  laplace: 'Pooled Laplace',
  'tabpfn-v2-512': 'Tabular transformer · V2 / 512',
  'logistic-point-512': 'Logistic · 512 labels',
  'gbdt-point-512': 'Boosted trees · 512 labels',
  'logistic-point-selected': 'Logistic · selected',
  'gbdt-point-selected': 'Boosted trees · selected',
};

const policyLabels: Record<string, string> = {
  incumbent: 'Incumbent', 'fixed-8': 'Fixed 8%', 'fixed-9': 'Fixed 9%', 'fixed-10': 'Fixed 10%',
  greedy: 'Greedy mean', cautious: 'Cautious improvement',
};

const modelName = (name: string) => modelLabels[name] ?? name.replaceAll('-', ' ');
const fixtureName = (name: string) => name.replaceAll('_', ' ');
const probabilityError = (value: number) => value.toFixed(4);

function ScenarioControl({ family, onChange }: { family: Family; onChange: (family: Family) => void }) {
  return <div className="audit-switch" role="group" aria-label="Simulation scenario">
    <button type="button" aria-pressed={family === 'smooth'} onClick={() => onChange('smooth')}>Smooth</button>
    <button type="button" aria-pressed={family === 'nonlinear'} onClick={() => onChange('nonlinear')}>Nonlinear challenge</button>
  </div>;
}

function CalibrationPlot({ score }: { score: Prediction }) {
  const points = score.calibration.filter((bin): bin is CalibrationBin & { prediction: number; observed: number } =>
    bin.count > 0 && bin.prediction !== null && bin.observed !== null,
  );
  const x = (value: number) => 52 + value * 286;
  const y = (value: number) => 234 - value * 198;
  return <>
    <svg className="audit-calibration-chart" viewBox="0 0 370 270" role="img" aria-label={`Calibration for ${modelName(score.model)}: mean predicted probability against observed acceptance in nonempty bins`}>
      {[0, 0.5, 1].map(tick => <g key={tick}>
        <line className="audit-chart-grid" x1={x(0)} x2={x(1)} y1={y(tick)} y2={y(tick)} />
        <text x={x(0) - 8} y={y(tick) + 3} textAnchor="end">{percent(tick)}</text>
        <text x={x(tick)} y={y(0) + 18} textAnchor="middle">{percent(tick)}</text>
      </g>)}
      <line className="audit-chart-ideal" x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} />
      {points.length > 1 && <polyline className="audit-chart-series" points={points.map(point => `${x(point.prediction)},${y(point.observed)}`).join(' ')} />}
      {points.map(point => <circle key={`${point.lower}-${point.upper}`} className="audit-chart-point" cx={x(point.prediction)} cy={y(point.observed)} r="4.5">
        <title>{`Predicted ${percent(point.prediction, 1)}, observed ${percent(point.observed, 1)}, ${point.count} responses`}</title>
      </circle>)}
      <text className="audit-chart-axis" x={x(0.5)} y="267" textAnchor="middle">Mean predicted probability</text>
      <text className="audit-chart-axis" transform="translate(12 135) rotate(-90)" textAnchor="middle">Observed acceptance</text>
    </svg>
    <div className="audit-legend"><span><i className="audit-legend-point" />Observed bins</span><span><i className="audit-legend-ideal" />Perfect agreement</span></div>
    <p className="audit-note">Each point is a nonempty bin; lines connect those points. Empty bins are omitted.</p>
    <details className="audit-disclosure"><summary>Inspect calibration bin counts</summary>
      <div className="audit-table-wrap"><table className="audit-table audit-compact"><caption className="audit-sr-only">Calibration bins for {modelName(score.model)}</caption><thead><tr><th scope="col">Predicted range</th><th scope="col">Count</th><th scope="col">Mean prediction</th><th scope="col">Observed</th></tr></thead>
        <tbody>{score.calibration.map(bin => <tr key={`${bin.lower}-${bin.upper}`}><th scope="row">{percent(bin.lower, 1)}–{percent(bin.upper, 1)}</th><td>{bin.count}</td><td>{bin.prediction === null ? '—' : percent(bin.prediction, 1)}</td><td>{bin.observed === null ? '—' : percent(bin.observed, 1)}</td></tr>)}</tbody>
      </table></div>
    </details>
  </>;
}

function PredictionPanel({ rows, reference = false }: { rows: Prediction[]; reference?: boolean }) {
  const titleId = reference ? "audit-reference-prediction-title" : "audit-prediction-title";
  const [selected, setSelected] = useState('laplace');
  const selectedScore = rows.find(row => row.model === selected) ?? rows[0];
  return <section className="audit-section" aria-labelledby={titleId}>
    <div className="audit-section-heading"><div><span className="eyebrow">01 / Prediction</span><h2 id={titleId}>How well do predictions hold up?</h2><p>Frozen models scored on the same fictional sealed-test applicants. Lower loss is better.</p></div><span className="audit-tag">Sealed test · {rows[0]?.n.toLocaleString() ?? '—'} applicants</span></div>
    <div className="audit-prediction-grid">
      <div className="panel audit-panel"><div className="audit-card-heading"><div><span className="eyebrow">Same factual outcomes</span><h3>Model comparison</h3></div></div>
        <div className="audit-table-wrap"><table className="audit-table"><caption className="audit-sr-only">Sealed-test prediction scores</caption><thead><tr><th scope="col">Method</th><th scope="col">Supported / unsupported / missing</th><th scope="col">Brier</th><th scope="col">Log loss</th><th scope="col">Calibration error</th></tr></thead><tbody>
          {rows.map(row => <tr key={row.model}><th scope="row">{modelName(row.model)}</th><td>{row.advisory_supported_n} / {row.advisory_unsupported_n} / {row.missing_prediction_n}<small>{row.n} scored / {row.cohort_n} total</small></td><td>{row.brier.toFixed(3)}</td><td>{row.log_loss.toFixed(3)}</td><td>{row.calibration_error.toFixed(3)}</td></tr>)}
        </tbody></table></div>
        <p className="audit-note">Support is the advisory-support-heuristic-v1 gate, separate from confidence. All scored cases remain in losses. Point scores describe predictive accuracy; they do not establish that uncertainty is reliable.</p>
      </div>
      <div className="panel audit-panel"><div className="audit-card-heading"><div><span className="eyebrow">Observed versus predicted</span><h3>Calibration by bin</h3></div><label className="audit-select-label">Model<select value={selectedScore?.model ?? ''} onChange={event => setSelected(event.target.value)}>{rows.map(row => <option value={row.model} key={row.model}>{modelName(row.model)}</option>)}</select></label></div>
        {selectedScore ? <><CalibrationPlot score={selectedScore} /><details className="audit-disclosure"><summary>Scores by rate and segment</summary><div className="audit-table-wrap"><table className="audit-table audit-compact"><caption className="audit-sr-only">Scores by rate and segment for {modelName(selectedScore.model)}</caption><thead><tr><th scope="col">Slice</th><th scope="col">Count</th><th scope="col">Brier</th><th scope="col">Log loss</th></tr></thead><tbody>{selectedScore.slices.map(slice => <tr key={slice.slice}><th scope="row">{slice.slice.replace(':', ' · ')}</th><td>{slice.n}</td><td>{slice.brier.toFixed(3)}</td><td>{slice.log_loss.toFixed(3)}</td></tr>)}</tbody></table></div></details></> : <p className="audit-note">No calibration data available for this scenario.</p>}
      </div>
    </div>
  </section>;
}

function NumericalPanel({ rows }: { rows: Evaluation['numerical'] }) {
  return <section className="audit-section" aria-labelledby="audit-numerical-title">
    <div className="audit-section-heading"><div><span className="eyebrow">02 / Numerical reference</span><h2 id="audit-numerical-title">A local approximation has limits.</h2><p>Independent 1D and 2D integrals compare the q05, q50 and q95 predictive quantiles with the Laplace approximation.</p></div><span className="audit-tag">Absolute error · threshold 0.03</span></div>
    <div className="panel audit-panel"><div className="audit-table-wrap"><table className="audit-table audit-reference-table"><caption className="audit-sr-only">Numerical integration reference results</caption><thead><tr><th scope="col">Fixture</th><th scope="col">Dimensions</th><th scope="col">Max error / tolerance</th><th scope="col">Grid change / tail bound</th><th scope="col">Result</th><th scope="col">Disposition</th></tr></thead><tbody>
      {rows.map(row => <tr key={row.fixture}><th scope="row"><span className="audit-capitalize">{fixtureName(row.fixture)}</span><small>{fixtureName(row.category)}</small></th><td>{row.dimensions}D</td><td className="audit-numeric">{probabilityError(row.max_quantile_error)} <span className="audit-dim">/ {row.tolerance.toFixed(2)}</span></td><td className="audit-numeric">{row.grid_quantile_difference.toExponential(2)}<small>tail ≤ {row.posterior_tail_bound.toExponential(2)}</small></td><td><span className={`audit-result ${row.status === 'pass' ? 'audit-result-pass' : 'audit-result-review'}`}>{row.status === 'pass' ? 'Within limit' : row.status === 'reference_unreliable' ? 'Reference uncertain' : 'Exceeds limit'}</span></td><td>{row.fallback ? <><strong className="audit-fallback">{fixtureName(row.fallback.status)} · {euro(row.fallback.immediate_review_cost_cents)} cost</strong><small>{row.disposition}</small></> : <span>{row.disposition}</span>}</td></tr>)}
    </tbody></table></div>
    <p className="audit-note">Grid change is an empirical refinement diagnostic, not a guaranteed quantile-error bound. The tail bound is conservative. Passing these selected fixtures does not validate a higher-dimensional posterior or automatically detect every problematic case.</p>
    </div>
  </section>;
}

function CoverageCell({ row }: { row: Coverage | undefined }) {
  if (!row) return <td>—</td>;
  return <td><strong>{percent(row.coverage, 1)} <span className="audit-dim">({row.covered}/{row.seeds})</span></strong><small>95% Wilson {percent(row.interval_low, 1)}–{percent(row.interval_high, 1)}</small><small>Mean interval width {(row.mean_width * 100).toFixed(1)} pp{row.target_standard_error > 0 ? ` · target SE ${row.target_standard_error.toFixed(4)}` : ''}</small></td>;
}

function CoveragePanel({ rows }: { rows: Coverage[] }) {
  const probes = [...new Set(rows.map(row => row.probe))];
  return <section className="audit-section" aria-labelledby="audit-coverage-title">
    <div className="audit-section-heading"><div><span className="eyebrow">03 / Uncertainty</span><h2 id="audit-coverage-title">Coverage depends on the question.</h2><p>Coverage of 90% posterior intervals over repeated fixed-world simulations, with finite-seed uncertainty shown for each probe.</p></div><span className="audit-tag">{rows[0]?.seeds ?? '—'} runs per probe</span></div>
    <div className="panel audit-panel"><div className="audit-targets"><div><strong>Exact Beta cells</strong><span>Cohort-average acceptance probability</span></div><div><strong>Pooled Laplace</strong><span>Acceptance conditional on the full visible context</span></div></div>
      <div className="audit-table-wrap"><table className="audit-table audit-coverage-table"><caption className="audit-sr-only">Repeated-data coverage by probe and model</caption><thead><tr><th scope="col">Probe / support</th><th scope="col">Exact Beta cells</th><th scope="col">Pooled Laplace</th></tr></thead><tbody>
        {probes.map(probe => { const beta = rows.find(row => row.probe === probe && row.model === 'beta'); const laplace = rows.find(row => row.probe === probe && row.model === 'laplace'); const support = beta?.support ?? laplace?.support ?? 'unknown'; return <tr key={probe}><th scope="row">{probe}<small className={`audit-support audit-support-${support}`}>{support === 'zero' ? 'Zero observed support' : `${support} support`}</small></th><CoverageCell row={beta} /><CoverageCell row={laplace} /></tr>; })}
      </tbody></table></div>
      <p className="audit-note">Wilson intervals describe uncertainty from the finite number of simulation runs, not a guarantee of nominal coverage. Mean interval widths are on the probability scale; target SE is shown where estimated. The zero-support cell remains ineligible for automated advice. This is not simulation-based calibration.</p>
    </div>
  </section>;
}

function PolicyPanel({ rows }: { rows: Evaluation['policies'] }) {
  return <section className="audit-section" aria-labelledby="audit-policy-title">
    <div className="audit-section-heading"><div><span className="eyebrow">04 / Decisions</span><h2 id="audit-policy-title">What do the policies trade off?</h2><p>Fictional evaluator-expected outcomes on the full sealed-test cohort, including manual reviews.</p></div><span className="audit-tag">All eligible applicants</span></div>
    <div className="panel audit-panel"><div className="audit-table-wrap"><table className="audit-table audit-policy-table"><caption className="audit-sr-only">Simulated policy outcomes on the full test cohort</caption><thead><tr><th scope="col">Model / policy</th><th scope="col">Expected acceptance<br />per applicant</th><th scope="col">Expected contribution<br />per applicant</th><th scope="col">Difference vs incumbent<br />95% paired bootstrap</th><th scope="col">Recommended / reviewed</th><th scope="col">Total review cost</th><th scope="col">Support violations</th></tr></thead><tbody>
      {rows.map(row => <tr key={`${row.model}-${row.policy}`}><th scope="row">{policyLabels[row.policy] ?? row.policy}<small>{modelName(row.model)}</small></th><td>{percent(row.expected_acceptance, 1)}</td><td>{euro(row.mean_expected_contribution_cents, 2)}</td><td>{euro(row.paired_difference_cents, 2)}<small>{euro(row.paired_bootstrap_low_cents, 2)} to {euro(row.paired_bootstrap_high_cents, 2)}</small></td><td>{percent(row.recommendation_coverage, 1)}<small>{row.reviews} reviews / {row.n.toLocaleString()} total</small></td><td>{euro(row.review_handling_cost_cents)}</td><td>{row.support_violations}</td></tr>)}
    </tbody></table></div><p className="audit-note">Mean contribution already includes review costs. Reviews receive no assumed later conversion. The paired applicant bootstrap is not a posterior interval; these figures are not commercial performance estimates.</p></div>
  </section>;
}

function findingData(finding: Finding): Record<string, unknown> | null {
  try {
    const detail: unknown = JSON.parse(finding.detail);
    return detail !== null && typeof detail === 'object' && !Array.isArray(detail) ? detail as Record<string, unknown> : null;
  } catch { return null; }
}

function FindingDetails({ finding }: { finding: Finding }) {
  const data = findingData(finding);
  if (!data) return <p>{finding.detail}</p>;
  const convergence = typeof data.converged === 'boolean' ? (data.converged ? 'Yes' : 'No') : 'Not reported';
  const optimizer = typeof data.optimizer_success === 'boolean' ? (data.optimizer_success ? 'Yes' : 'No') : 'Not reported';
  const condition = typeof data.condition_number === 'number' ? data.condition_number.toFixed(1) : '—';
  const gradient = typeof data.gradient_norm === 'number' ? data.gradient_norm.toExponential(2) : '—';
  return <><div className="audit-finding-metrics"><span>Convergence gate <strong>{convergence}</strong></span><span>Optimizer success <strong className={data.optimizer_success === false ? 'audit-warning-text' : ''}>{optimizer}</strong></span><span>Condition number <strong>{condition}</strong></span><span>Gradient norm <strong>{gradient}</strong></span></div>{typeof data.message === 'string' && <p className="audit-finding-message">Optimizer: {data.message}</p>}{typeof data.reference_check === 'string' && <p className="audit-finding-message">This fit's reference check: {fixtureName(data.reference_check)}.</p>}</>;
}

function PriorPanel({ rows, findings }: { rows: Prediction[]; findings: Finding[] }) {
  const optimizerWarnings = findings.filter(finding => findingData(finding)?.optimizer_success === false).length;
  return <section className="audit-section" aria-labelledby="audit-prior-title">
    <div className="audit-section-heading"><div><span className="eyebrow">05 / Assumptions</span><h2 id="audit-prior-title">Inspect the prior and the fit.</h2><p>Prior presets were fixed before test scoring; small differences are shown at extra precision.</p></div></div>
    <div className="audit-prior-grid"><div className="panel audit-panel"><div className="audit-card-heading"><div><span className="eyebrow">Pooled Laplace · sealed test</span><h3>Prior sensitivity</h3></div></div><div className="audit-table-wrap"><table className="audit-table"><caption className="audit-sr-only">Pooled-model prior sensitivity scores</caption><thead><tr><th scope="col">Prior</th><th scope="col">Supported / unsupported / missing</th><th scope="col">Brier</th><th scope="col">Log loss</th><th scope="col">Calibration error</th></tr></thead><tbody>
      {rows.map(row => <tr key={row.model}><th scope="row" className="audit-capitalize">{row.model.replace('laplace-prior-', '')}</th><td>{row.advisory_supported_n} / {row.advisory_unsupported_n} / {row.missing_prediction_n}<small>{row.n} scored / {row.cohort_n} total</small></td><td>{row.brier.toFixed(5)}</td><td>{row.log_loss.toFixed(5)}</td><td>{row.calibration_error.toFixed(5)}</td></tr>)}
    </tbody></table></div><p className="audit-note">These are point-prediction scores under each prior, not a measurement of every posterior interval's sensitivity.</p></div>
      <div className="panel audit-panel"><div className="audit-card-heading"><div><span className="eyebrow">Fit diagnostics</span><h3>Findings</h3></div></div>{optimizerWarnings > 0 && <p className="audit-finding-callout">{optimizerWarnings} of {findings.length} fits report optimizer success as false. Open a finding for the message and numerical diagnostics.</p>}<div className="audit-findings">{findings.map(finding => <details className="audit-finding" key={finding.name}><summary><span>{finding.name}</span><span className="audit-tag">{finding.status}</span></summary><FindingDetails finding={finding} /></details>)}</div><p className="audit-note">A convergence gate and an optimizer's success flag are separate diagnostics. Neither certifies the approximation.</p></div>
    </div>
  </section>;
}

function AdversePanel({ report }: { report: Evaluation }) {
  const findings = report.findings.filter(f => !f.name.includes(' · '));
  const rows = report.prediction.filter(p => p.cohort.includes('challenge'));
  return <section className="audit-section" aria-labelledby="audit-adverse-title">
    <div className="audit-section-heading"><div><span className="eyebrow">06 / Changing conditions</span><h2 id="audit-adverse-title">Separate a signal from a conclusion.</h2><p>New menus, selected feedback and changed response mechanisms test different assumptions.</p></div></div>
    <div className="panel audit-panel"><div className="audit-findings">{findings.map(finding => <details className="audit-finding" key={finding.name}><summary><span>{finding.name}</span><span className="audit-tag">{finding.status}</span></summary><FindingDetails finding={finding} /></details>)}</div>
      <div className="audit-table-wrap"><table className="audit-table"><caption>Frozen-model challenge and point-only recalibration</caption><thead><tr><th scope="col">Challenge / method</th><th scope="col">N</th><th scope="col">Supported / unsupported / missing</th><th scope="col">Brier</th><th scope="col">Log loss</th><th scope="col">Calibration error</th></tr></thead><tbody>{rows.map(row => <tr key={`${row.family}-${row.model}`}><th scope="row">{row.family}<small>{modelName(row.model)}</small></th><td>{row.n}</td><td>{row.advisory_supported_n} / {row.advisory_unsupported_n} / {row.missing_prediction_n}<small>{row.n} scored / {row.cohort_n} total</small></td><td>{row.brier.toFixed(4)}</td><td>{row.log_loss.toFixed(4)}</td><td>{row.calibration_error.toFixed(4)}</td></tr>)}</tbody></table></div><p className="audit-note">The recalibrated comparator is a point prediction only. The primary posterior draws are unchanged, and calibration cannot create exposure for an unsupported rate.</p>
    </div>
  </section>;
}

export function ModelAudit({ report }: { report: Evaluation }) {
  const [family, setFamily] = useState<Family>('smooth');
  const [view, setView] = useState<ComparisonView>('same512');
  const familyPredictions = report.prediction.filter(row => row.family === family && row.cohort === 'sealed test');
  const predictions = familyPredictions.filter(row => report.comparison ? comparisonModels(view).includes(row.model) : !row.model.startsWith('laplace-prior-'));
  const references = familyPredictions.filter(row => ['beta', 'laplace', 'logistic-point', 'empirical-rate', 'empirical-cell'].includes(row.model));
  const priors = familyPredictions.filter(row => row.model.startsWith('laplace-prior-'));
  const coverage = report.coverage.filter(row => row.family === family);
  const policies = report.policies.filter(row => row.family === family);
  const findings = report.findings.filter(finding => finding.name.startsWith(`${family} ·`) && finding.name.includes('MAP diagnostics'));
  const transformer = predictions.some(p => p.model === 'tabpfn-v2-512');
  const passed = report.numerical.filter(row => row.status === 'pass').length;
  return <>
    <div className="page-heading audit-page-heading"><div><div className="eyebrow">04 / Model audit</div><h1>Look closer at the evidence.</h1><p>Compare prediction, uncertainty and decisions in a fictional simulation. Each answers a different question.</p></div><button className="button secondary" type="button" onClick={() => downloadEvidence(report)}>Export audit evidence ↓</button></div>
    <div className="audit-toolbar"><div><span className="eyebrow">Simulation scenario</span><p>{family === 'smooth' ? 'A smooth, well-specified response pattern.' : 'Correlated business attributes and an exactly marginalized urgency / competing-offer mixture. Authored assumptions, not measured customer behavior.'}</p></div><ScenarioControl family={family} onChange={setFamily} /></div>
    {transformer && <div className="panel transformer-note"><div><span className="eyebrow">Real tabular transformer challenger</span><h2>Pretrained V2 · 512 factual context rows · CPU</h2><p>Point prediction only. Model uncertainty is not estimated; these outputs are excluded from P5 and Thompson sampling. Fixed logistic and boosted-tree controls use the same 512 labels. Shared full-fit history and external pretraining are disclosed below. Built with PriorLabs-TabPFN.</p></div><a className="button secondary" href="/api/challenger" target="_blank" rel="noreferrer">Open benchmark manifest ↗</a></div>}
    <div className="audit-summary"><div><span>Sealed-test applicants</span><strong>{predictions[0]?.n.toLocaleString() ?? '—'}</strong><small>Same factual cohort per model</small></div><div><span>Numerical fixtures within limit</span><strong>{passed}<em> / {report.numerical.length}</em></strong><small>Selected low-dimensional cases</small></div><div><span>Repeated-data runs</span><strong>{coverage[0]?.seeds ?? '—'}</strong><small>Per probe · finite-seed uncertainty</small></div></div>
    {report.comparison && <div className="audit-toolbar"><div><span className="eyebrow">Point model comparison</span><p>Three main families; Exact Beta remains a separate counting reference.</p></div><div className="audit-switch" role="group" aria-label="Comparison budget"><button type="button" aria-pressed={view === 'same512'} onClick={() => setView('same512')}>Same 512 fitting labels</button><button type="button" aria-pressed={view === 'development'} onClick={() => setView('development')}>Full fit + development</button></div></div>}
    <PredictionPanel rows={predictions} />
    {report.comparison && <PointDevelopment comparison={report.comparison} family={family} view={view} />}
    {report.comparison && <details className="audit-disclosure"><summary>Exact Beta counting reference and original full-fit controls</summary><p>These references use {report.datasets.find(d => d.family === family)?.fit} fitting labels with fixed settings. Exact Beta is a counting reference; the base C=1 logistic fit reuses its identical search configuration without extra fitting. Selection labels do not choose these reference settings.</p><PredictionPanel rows={references} reference /></details>}
    <NumericalPanel rows={report.numerical} />
    <CoveragePanel rows={coverage} />
    <PolicyPanel rows={policies} />
    <PriorPanel rows={priors} findings={findings} />
    <AdversePanel report={report} />
    <aside className="audit-limits"><span className="eyebrow">Read these results in context</span><ul>{report.interpretation.map(item => <li key={item}>{item}</li>)}</ul><p>Saved run <code>{report.provenance.run_id}</code> · authored fictional simulation</p></aside>
  </>;
}
