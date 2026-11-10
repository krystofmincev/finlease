import type { Decision, Offer, Request } from '../api/client';
import { downloadEvidence } from '../api/client';
import { Interval } from '../components/Interval';
import { euro, percent, reasonText, segmentName } from '../components/format';

type Props = { request: Request; update: (value: Request) => void; decision: Decision | null; busy: boolean; error: string | null; methods: string[]; presets: Request['context'][] };

function OfferCard({ offer, decision }: { offer: Offer; decision: Decision }) {
  const chosen = decision.cautious.offer_id === offer.offer_id;
  return <article className={`offer-card ${chosen ? 'selected' : ''}`}>
    <div className="offer-top"><span className="eyebrow">Nominal annual rate</span>{offer.offer_id === decision.incumbent_offer_id && <span className="pill neutral">Incumbent</span>}</div>
    <div className="rate">{(offer.rate_bps / 100).toFixed(1)}<span>%</span></div>
    <div className="payment">{euro(offer.monthly_payment_cents, 2)}<span> / month</span></div>
    <div className="total">{euro(offer.total_repayment_cents)} total · includes balloon</div>
    <div className="card-divider" />
    <div className="metric-row"><span>Acceptance estimate</span><strong>{percent(offer.acceptance.mean, 1)}</strong></div>
    <Interval low={offer.acceptance.q05} high={offer.acceptance.q95} mean={offer.acceptance.mean} label={`90% model interval ${percent(offer.acceptance.q05)} to ${percent(offer.acceptance.q95)}`} />
    <div className="interval-labels"><span>{percent(offer.acceptance.q05)}</span><span>90% model interval</span><span>{percent(offer.acceptance.q95)}</span></div>
    <div className="metric-row contribution"><span>Expected contribution</span><strong>{euro(offer.expected_contribution_cents.mean)}</strong></div>
    <div className="support-row"><span className={`support-dot ${offer.support.state}`} /><span>{offer.support.response_count.toLocaleString()} responses</span><span className={`support-label ${offer.support.state}`}>{offer.support.automated ? 'Supported' : offer.support.state === 'prior_only' ? 'Prior only' : 'Review'}</span></div>
    {offer.support.reasons.length > 0 && <p className="support-reasons">{offer.support.reasons.map(r => r.replaceAll('_', ' ')).join(' · ')}</p>}
    {chosen && <div className="selected-label">✓ Cautious policy choice</div>}
  </article>;
}

export function CaseReview({ request, update, decision, busy, error, methods, presets }: Props) {
  const context = request.context;
  function contextChange(key: keyof Request['context'], value: string | number) {
    if (key === 'segment') {
      const preset = presets.find(p => p.segment === value);
      if (preset) { update({ ...request, context: preset }); return; }
    }
    update({ ...request, context: { ...context, [key]: value } });
  }
  const choice = decision?.cautious;
  const chosen = decision?.offers.find(o => o.offer_id === choice?.offer_id);
  return <>
    <div className="page-heading"><div><div className="eyebrow">01 / Case review</div><h1>Evidence before a change.</h1><p>Should the adviser keep the current lease offer or change it? Compare customer payments, expected contribution and the evidence behind the choice.</p></div><button className="button secondary" disabled={!decision || busy} onClick={() => decision && downloadEvidence(decision)}>↓ Export evidence</button></div>
    <div className="case-layout">
      <aside className="panel case-inputs">
        <div className="panel-title"><h2>The fictional case</h2><span className="pill neutral">Pre-approved</span></div>
        <p className="muted small">Inputs known before the offer. Every response is simulated.</p>
        <label>Business stage<select value={context.segment} onChange={e => contextChange('segment', e.target.value)}>{Object.entries(segmentName).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label>
        <label>Business age <span className="label-value">{context.business_age_years.toFixed(1)} years</span><input type="range" min="0.5" max="30" step="0.5" value={context.business_age_years} onChange={e => contextChange('business_age_years', Number(e.target.value))} /></label>
        <label>Monthly turnover · €<input type="number" min="100" max="2000000" step="1000" value={context.monthly_turnover_cents / 100} onChange={e => contextChange('monthly_turnover_cents', Math.round(Number(e.target.value) * 100))} /></label>
        <label>Existing debt service / turnover <span className="label-value">{percent(context.existing_debt_burden)}</span><input type="range" min="0" max="0.85" step="0.01" value={context.existing_debt_burden} onChange={e => contextChange('existing_debt_burden', Number(e.target.value))} /><span className="field-note">Existing burden excludes the candidate lease payment.</span></label>
        <div className="card-divider" />
        <label>Rate card<select value={request.card_id} onChange={e => update({ ...request, card_id: e.target.value as Request['card_id'], as_of: e.target.value === 'nov-2026' ? '2026-11-09' : '2026-12-09' })}><option value="nov-2026">November 2026 · observed</option><option value="dec-2026">December 2026 · new menu</option></select></label>
        <label>Inference model<select value={request.method} onChange={e => update({ ...request, method: e.target.value as Request['method'] })}>{methods.map(m => <option key={m} value={m}>{m === 'beta' ? 'Exact Beta cells' : 'Pooled logistic · Laplace'}</option>)}</select></label>
        <label>Prior sensitivity<select value={request.prior} onChange={e => update({ ...request, prior: e.target.value as Request['prior'] })}><option value="default">Default prior</option><option value="weak">Weaker prior</option><option value="strong">Stronger prior</option></select></label>
        <details><summary>Financing assumptions</summary><div className="assumptions"><label>Financed amount · €<input type="number" min="1" max="1000000" value={request.finance.principal_cents / 100} onChange={e => update({ ...request, finance: { ...request.finance, principal_cents: Math.round(Number(e.target.value) * 100) } })} /></label><p>{request.finance.term_months} monthly payments, plus {euro(request.finance.balloon_cents)} with the last payment. The final instalment adjusts for cent rounding. Inference is supported only for the trained terms.</p></div></details>
      </aside>
      <section className={`case-results ${busy ? 'calculating' : ''}`} aria-busy={busy}>
        {error && <div className="error" role="alert">{error}</div>}
        {decision && <>
          <div className={`decision-banner ${choice?.status === 'review' ? 'needs-review' : ''}`}>
            <div className="decision-icon">{choice?.status === 'review' ? '?' : '↗'}</div><div><div className="eyebrow">Cautious improvement policy</div><h2>{choice?.status === 'review' ? 'Pause for manual review.' : choice?.status === 'retain' ? `Keep the incumbent at ${chosen ? chosen.rate_bps / 100 : '—'}%.` : `Evidence supports ${chosen ? chosen.rate_bps / 100 : '—'}%.`}</h2><p>{choice && reasonText(choice.reason)}</p></div>
            <span className="pill decision-pill">{busy ? 'Calculating…' : choice?.status === 'review' ? 'Review required' : 'Supported decision'}</span>
          </div>
          <div className="section-line"><h2>Compare the eligible offers</h2><span>Same case · same parameter draws</span></div>
          <div className="offer-grid">{decision.offers.map(offer => <OfferCard key={offer.offer_id} offer={offer} decision={decision} />)}</div>
          <div className="panel reasoning-panel"><div className="section-line"><h2>A point estimate is one part of the decision.</h2><span className="pill neutral">Expected value, not outcome noise</span></div>
            <div className="comparison-grid"><div><span className="eyebrow">Greedy mean policy</span><h3>{decision.greedy.offer_id ? `${Number(decision.greedy.offer_id.slice(1)) / 100}%` : 'Manual review'}</h3><p>{reasonText(decision.greedy.reason)}</p></div><div><span className="eyebrow">Cautious policy</span><h3>{choice?.offer_id ? `${Number(choice.offer_id.slice(1)) / 100}%` : 'Manual review'}</h3><p>Maximise the lower bound of paired improvement, then require more than {euro(decision.improvement_margin_cents)}.</p></div></div>
            <div className="improvement-table"><div className="table-head"><span>Offer vs incumbent</span><span>Mean improvement</span><span>5th percentile</span></div>{decision.offers.map(o => <div className="table-row" key={o.offer_id}><span>{o.rate_bps / 100}% {o.offer_id === decision.incumbent_offer_id && <small>· incumbent</small>}</span><strong>{o.improvement_cents ? euro(o.improvement_cents.mean) : 'Unavailable'}</strong><strong className={o.improvement_cents && o.improvement_cents.q05 > decision.improvement_margin_cents ? 'positive' : ''}>{o.improvement_cents ? euro(o.improvement_cents.q05) : 'Unavailable'}</strong></div>)}</div>
            <p className="footnote">These bounds concern expected contribution under the model. They are not a guarantee for an individual customer or a repeated-selection guarantee.</p>
          </div>
          <details className="panel evidence-drawer"><summary>Calculation & evidence details <span>{decision.draw_count.toLocaleString()} joint draws</span></summary><div className="drawer-body"><p>{decision.uncertainty_method} · {decision.uncertainty_target}</p><p>{decision.quantile_method}. Seed {request.draw_seed}; numerical review within {euro(decision.numerical_tolerance_cents)} of the margin.</p>{decision.diagnostics.map(d => <p key={d}>{d}</p>)}<p>Fictional contribution if completed: {decision.offers.map(o => `${o.rate_bps / 100}% → ${euro(o.contribution_if_accepted_cents)}`).join('; ')}. Pre-decision cost {euro(decision.offers[0].decision_cost_cents)} per case. Manual review earns no financing income and costs €15.</p><code>{decision.provenance.run_id}</code><span className="latency">{decision.elapsed_ms.toFixed(1)} ms inference</span></div></details>
        </>}
      </section>
    </div>
  </>;
}
