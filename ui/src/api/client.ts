import Ajv2020 from 'ajv/dist/2020';
import type { ValidateFunction } from 'ajv';
import schema from './schemas.json';
import type { components } from './generated';

export type Metadata = components['schemas']['Metadata'];
export type Decision = components['schemas']['DecisionResult'];
export type Request = Metadata['default_request'];
export type Context = Request['context'];
export type Offer = Decision['offers'][number];
export type Evaluation = components['schemas']['EvaluationReport'];
export type Replays = components['schemas']['ReplayReport'];

const ajv = new Ajv2020({ strict: true, allErrors: true });
function validator<T>(name: string): ValidateFunction<T> {
  return ajv.compile<T>({ ...schema, $ref: `#/$defs/${name}` });
}
const metaValidator = validator<Metadata>('Metadata');
const decisionValidator = validator<Decision>('DecisionResult');
const reportValidator = validator<Evaluation>('EvaluationReport');
const replayValidator = validator<Replays>('ReplayReport');

async function load<T>(path: string, validate: ValidateFunction<T>, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(`The calculation could not be completed (${response.status}). Check the case inputs.`);
  const value: unknown = await response.json();
  if (!validate(value)) throw new Error(`Response contract failed: ${ajv.errorsText(validate.errors)}`);
  return value;
}

export const getMetadata = (signal?: AbortSignal) => load('/api/meta', metaValidator, { signal });
export const getReport = (signal?: AbortSignal) => load('/api/report', reportValidator, { signal });
export const getReplays = (signal?: AbortSignal) => load('/api/replays', replayValidator, { signal });
export const getDecision = (request: Request, signal?: AbortSignal) => load('/api/decision', decisionValidator, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request), signal,
});

export function downloadEvidence(value: Decision | Metadata | Evaluation | Replays) {
  const href = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = href; link.download = `lease-lab-${value.provenance.run_id}.json`; link.click();
  URL.revokeObjectURL(href);
}
