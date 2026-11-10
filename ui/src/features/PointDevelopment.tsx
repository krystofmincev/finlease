import type { Evaluation } from '../api/client';

type Comparison = NonNullable<Evaluation['comparison']>;
type Method = Comparison['methods'][number];
export type ComparisonView = 'same512' | 'development';
export const comparisonModels = (view: ComparisonView) => view === 'same512'
  ? ['logistic-point-512', 'gbdt-point-512', 'tabpfn-v2-512']
  : ['logistic-point-selected', 'gbdt-point-selected', 'tabpfn-v2-512'];

function LossPlot({ train, development, x, label, chosen }: {
  train: number[]; development: number[]; x: number[]; label: string; chosen?: number;
}) {
  const minimum = Math.min(...train, ...development) * .95;
  const maximum = Math.max(...train, ...development) * 1.05;
  const px = (value: number) => 52 + (value - x[0]) / Math.max(1, x[x.length - 1] - x[0]) * 300;
  const py = (value: number) => 210 - (value - minimum) / Math.max(.001, maximum - minimum) * 170;
  return <svg viewBox="0 0 390 260" className="audit-calibration-chart" role="img" aria-label={label}>
    {[minimum, (minimum + maximum) / 2, maximum].map(tick => <g key={tick}><line className="audit-chart-grid" x1="52" x2="352" y1={py(tick)} y2={py(tick)} /><text x="46" y={py(tick) + 4} textAnchor="end">{tick.toFixed(2)}</text></g>)}
    {[x[0], x[Math.floor(x.length / 2)], x[x.length - 1]].map((tick, i) => <text key={i} x={px(tick)} y="232" textAnchor="middle">{tick}</text>)}
    <polyline fill="none" className="audit-chart-series" points={train.map((v, i) => `${px(x[i])},${py(v)}`).join(' ')} />
    <polyline fill="none" className="audit-chart-ideal" points={development.map((v, i) => `${px(x[i])},${py(v)}`).join(' ')} />
    {chosen !== undefined && <line x1={px(chosen)} x2={px(chosen)} y1="40" y2="210" stroke="currentColor" strokeDasharray="3 5"><title>Chosen positive round {chosen}</title></line>}
    <text className="audit-chart-axis" x="202" y="255">{label}</text><text className="audit-chart-axis" transform="translate(12 135) rotate(-90)" textAnchor="middle">Factual log loss</text>
  </svg>;
}

function DevelopmentCurves({ method }: { method: Method }) {
  const trial = method.trials.find(t => t.candidate_id === method.chosen_candidate);
  return <div className="panel audit-panel"><h3>{method.model.startsWith('gbdt') ? 'Gradient-boosted trees' : 'Logistic regression'} · {method.feature.design}</h3>
    <p className="audit-note">{method.optimizer}. Chosen {method.chosen_candidate}; shared full-fit preprocessing. Curves condition on selection and use the early-stop development cohort, not final data.</p>
    <LossPlot train={method.curves.map(c => c.train_log_loss)} development={method.curves.map(c => c.early_stop_log_loss)} x={method.curves.map(c => c.fit_n)} label="Model-fitting labels" />
    <p className="audit-note">Solid: train · dashed: early-stop development. Logistic records solver iterations; these are label-budget curves, not epoch histories.</p>
    <table className="audit-table audit-compact"><caption>Actual label-budget observations</caption><thead><tr><th>Fit labels</th><th>Train loss</th><th>Development loss</th><th>Iterations / rounds</th></tr></thead><tbody>{method.curves.map(c => <tr key={c.fit_n}><td>{c.fit_n}</td><td>{c.train_log_loss.toFixed(5)}</td><td>{c.early_stop_log_loss.toFixed(5)}</td><td>{c.iterations}</td></tr>)}</tbody></table>
    {trial && trial.development_loss.length > 0 && <details className="audit-disclosure"><summary>Actual native tree stopping curve</summary>
      <LossPlot train={trial.train_loss} development={trial.development_loss} x={trial.train_loss.map((_, i) => i)} label="Boosting round (includes round zero)" chosen={trial.chosen_round ?? undefined} />
      <p>Chosen positive round {trial.chosen_round}; stopped at {trial.stopped_round}. {trial.reason}. Patience {String(trial.parameters.n_iter_no_change)}; tolerance {String(trial.parameters.tol)}. Native validation window; fixed-round refit verified against the selected prefix.</p>
    </details>}
  </div>;
}

export function PointDevelopment({ comparison, family, view }: { comparison: Comparison; family: 'smooth' | 'nonlinear'; view: ComparisonView }) {
  const methods = comparison.methods.filter(m => m.family === family);
  const displayed = methods.filter(m => comparisonModels(view).includes(m.model));
  const selected = methods.filter(m => m.selection === 'selected');
  return <section className="audit-section" aria-label="Point model development evidence">
    <div className="audit-section-heading"><div><span className="eyebrow">Development and cost</span><h2>Count the evidence used.</h2><p>Scenario: {comparison.scenario_versions[family]}</p><p>{view === 'same512' ? 'Identical 512 model-fitting IDs and base float32 matrices; no search, stopping or calibration labels.' : 'Selected full-fit methods use additional selection and stopping labels. The pretrained row retains its fixed 512-label budget.'}</p></div><span className="audit-tag">{comparison.status.replaceAll('_', ' ')}</span></div>
    <div className="panel audit-panel"><div className="audit-table-wrap"><table className="audit-table"><caption>Label roles and measured method costs</caption><thead><tr><th>Method / features</th><th>Fit / stop / select / calibrate</th><th>Monitor / consumed union / observed union</th><th>Shared history</th><th>Fit / prediction seconds</th></tr></thead><tbody>{displayed.map(m => <tr key={m.model}><th>{m.model}<small>{m.feature.design} · {m.feature.columns.length} columns · {m.feature.precision}</small></th><td>{m.budget.fit.n} / {m.budget.early_stop.n} / {m.budget.selection.n} / {m.budget.calibration.n}</td><td>{m.budget.monitoring.n} / {m.budget.consumed_union_n} / {m.budget.observed_union_n}<small>{m.budget.overlap.join('; ') || 'No role overlap'}</small></td><td>{m.budget.preprocessing_context_n} preprocessing contexts / {m.budget.support_history_n} support responses<small>{m.budget.external_pretraining ? 'External pretraining labels/cost not equalized' : 'No external pretraining'}</small></td><td>{m.fit_seconds.toFixed(3)} / {m.prediction_seconds.toFixed(3)}</td></tr>)}</tbody></table></div>
      <p className="audit-note">This conditions on common 3,600-row preprocessing and advisory history. It does not represent a complete deployment with only 512 observations. {comparison.development_fit_count} added sklearn fits across both families in {comparison.development_seconds.toFixed(3)}s; two full-fit reference configurations reuse search fits. Selected fit costs cover the winning fit; full search and curve costs are included in development totals and trial records. Prediction costs include frozen scoring and policy evaluation; pretraining cost is external.</p>
    </div>
    <details className="audit-disclosure"><summary>Inspect bounded selection, matched ablations and actual curves</summary>
      <p>Selection uses factual log loss only. All 9 logistic and 5 tree candidates per family are shown; compare matching C or learning rate/leaves to inspect feature effects. Smaller-label curves reuse selected settings without re-stopping. Calibration labels are zero for this comparison; the separate response-change Platt example uses 900.</p>
      <div className="audit-prediction-grid">{selected.map(m => <DevelopmentCurves key={m.model} method={m} />)}</div>
      <div className="audit-table-wrap"><table className="audit-table audit-compact"><caption>Complete trials and matched feature ablations · {family}</caption><thead><tr><th>Candidate / design</th><th>C or rate / leaves</th><th>Selection log loss / Brier / calibration error</th><th>Iterations / chosen / stopped</th><th>Status / seconds</th></tr></thead><tbody>{selected.flatMap(m => m.trials.map(t => <tr key={t.candidate_id}><th>{t.candidate_id}{t.candidate_id === m.chosen_candidate ? ' · selected' : ''}<small>{t.design} · {t.columns} columns</small></th><td>{t.parameters.C !== undefined ? `C=${String(t.parameters.C)}` : `${String(t.parameters.learning_rate)} / ${String(t.parameters.max_leaf_nodes)}`}</td><td>{t.selection_log_loss?.toFixed(5) ?? '—'} / {t.selection_brier?.toFixed(5) ?? '—'} / {t.selection_calibration_error?.toFixed(5) ?? '—'}</td><td>{t.iterations} / {t.chosen_round ?? '—'} / {t.stopped_round ?? '—'}</td><td>{t.status} · {t.elapsed_seconds.toFixed(3)}s<small>{t.failure ?? t.reason}</small></td></tr>))}</tbody></table></div>
      <p className="audit-note">Role IDs, transform/column identities, every trial curve and the stable selection commitment are retained in exported JSON. Freeze {comparison.freeze_sha256}; final recipe {comparison.final_seed} / {comparison.final_namespace}. Previously exposed report-only diagnostic data are not a new reserve.</p>
    </details>
  </section>;
}
