import { useEffect, useState } from 'react';
import type { Context, Decision, Evaluation, Metadata, Replays, Request } from './api/client';
import { getDecision, getMetadata, getReport, getReplays } from './api/client';
import { CaseReview } from './features/CaseReview';
import { EvidenceMap } from './features/EvidenceMap';
import { ModelAudit } from './features/ModelAudit';
import { LearningReplay } from './features/LearningReplay';

type View = 'case' | 'evidence' | 'learning' | 'audit';
const views: { id: View; name: string; description: string; number: string }[] = [
  { id: 'case', name: 'Case review', description: 'Choose with evidence', number: '01' },
  { id: 'evidence', name: 'Evidence map', description: 'Inspect observed support', number: '02' },
  { id: 'learning', name: 'Learning replay', description: 'Explore, then learn', number: '03' },
  { id: 'audit', name: 'Model audit', description: 'Challenge assumptions', number: '04' },
];

function Workspace({ metadata }: { metadata: Metadata }) {
  const [view, setView] = useState<View>('case');
  const [request, setRequest] = useState<Request>(() => ({
    ...metadata.default_request,
    context: metadata.case_presets.find(p => p.segment === 'growing') ?? metadata.default_request.context,
  }));
  const [calculation, setCalculation] = useState<{ forRequest: Request; result: Decision } | null>(null);
  const decision = calculation?.forRequest === request ? calculation.result : null;
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<Evaluation | null>(null);
  const [replays, setReplays] = useState<Replays | null>(null);
  const [chapterError, setChapterError] = useState<string | null>(null);
  function sameRun<T extends { provenance: Metadata['provenance'] }>(value: T): T {
    if (value.provenance.run_id !== metadata.provenance.run_id || value.provenance.config_sha256 !== metadata.provenance.config_sha256 || value.provenance.data_sha256 !== metadata.provenance.data_sha256) {
      throw new Error('The frozen source run changed. Reload the workspace before comparing evidence.');
    }
    return value;
  }
  useEffect(() => { window.scrollTo(0, 0); }, [view]);
  useEffect(() => {
    const controller = new AbortController();
    setChapterError(null);
    const failure = (cause: unknown) => { if (!controller.signal.aborted) setChapterError(cause instanceof Error ? cause.message : 'Could not load the saved chapter.'); };
    if (view === 'audit' && !report) getReport(controller.signal).then(sameRun).then(setReport).catch(failure);
    if (view === 'learning' && !replays) getReplays(controller.signal).then(sameRun).then(setReplays).catch(failure);
    return () => controller.abort();
  }, [view, report, replays]);
  useEffect(() => {
    const controller = new AbortController();
    setBusy(true); setError(null);
    setCalculation(null);
    const timer = window.setTimeout(() => {
      getDecision(request, controller.signal).then(sameRun).then(result => { if (!controller.signal.aborted) setCalculation({ forRequest: request, result }); }).catch((cause: unknown) => {
        if (!controller.signal.aborted) { setCalculation(null); setError(cause instanceof Error ? cause.message : 'Calculation failed.'); }
      }).finally(() => { if (!controller.signal.aborted) setBusy(false); });
    }, 180);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [request]);
  const selectSegment = (segment: Context['segment']) => setRequest(previous => ({ ...previous, context: metadata.case_presets.find(p => p.segment === segment) ?? { ...previous.context, segment } }));
  return <div className="app-shell"><aside className="sidebar"><a className="brand" href="#" onClick={() => setView('case')}><span className="brand-symbol">↗</span><span>LEASE<span className="brand-sub">Decision lab</span></span></a><div className="workspace-label">THE ADVISER WORKSPACE</div><nav aria-label="Main navigation">{views.map(item => <button key={item.id} className={`nav-item ${view === item.id ? 'active' : ''}`} onClick={() => setView(item.id)}><span className="nav-number">{item.number}</span><span><strong>{item.name}</strong><small>{item.description}</small></span>{view === item.id && <span className="nav-arrow">↗</span>}</button>)}</nav><div className="sidebar-bottom"><span className="live-dot" />Local scientific workspace<p>Fictional customers.<br />Real calculations.</p><div className="sidebar-version">Blueprint 1.1 <span>·</span> EUR / monthly</div></div></aside><div className="workspace"><header className="topbar"><div><span className="breadcrumb">Decision lab</span><span className="crumb-separator">/</span>{views.find(v => v.id === view)?.name}</div><div className="topbar-right"><span className="simulation-badge"><span />SIMULATION ONLY</span><span className="avatar">KL</span></div></header><main>
    {view === 'case' && <CaseReview request={request} update={setRequest} decision={decision} busy={busy} error={error} methods={metadata.available_methods} presets={metadata.case_presets} />}
    {view === 'evidence' && <EvidenceMap metadata={metadata} context={request.context} selectSegment={selectSegment} />}
    {view === 'audit' && report && <ModelAudit report={report} />}
    {view === 'learning' && replays && <LearningReplay report={replays} />}
    {((view === 'audit' && !report) || (view === 'learning' && !replays)) && <div className="panel pending-chapter"><span className="eyebrow">Saved offline evidence</span><h1>{view === 'learning' ? 'Learning replay' : 'Model audit'}</h1><p>{chapterError ?? 'Loading and validating the saved experiment…'}</p></div>}
    <footer>Authored synthetic data · Fictional economics · No real lending or approval advice <span>Pretrained transformer: Built with PriorLabs-TabPFN · {metadata.provenance.run_id}</span></footer>
  </main></div></div>;
}

export default function App() {
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    getMetadata(controller.signal).then(setMetadata).catch((cause: unknown) => { if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : 'Unable to load the lab.'); });
    return () => controller.abort();
  }, []);
  return metadata ? <Workspace metadata={metadata} /> : <div className="initial-state"><div className="brand-symbol">↗</div><h1>{error ? 'The lab could not load.' : 'Opening the decision lab…'}</h1><p>{error ?? 'Loading the frozen scientific bundle.'}</p>{error && <button className="button" onClick={() => location.reload()}>Try again</button>}</div>;
}
