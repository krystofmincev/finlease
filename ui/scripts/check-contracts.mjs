import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import Ajv2020 from 'ajv/dist/2020.js';

const schema=JSON.parse(readFileSync(new URL('../src/api/schemas.json',import.meta.url),'utf8'));
const fixtures=JSON.parse(readFileSync(new URL('../../.local/contract-fixtures.json',import.meta.url),'utf8'));
const ajv=new Ajv2020({strict:true,allErrors:true});
for (const [name,value] of Object.entries(fixtures)) {
  const validate=ajv.compile({...schema,$ref:`#/$defs/${name}`});
  assert(validate(value),ajv.errorsText(validate.errors));
  const missing=structuredClone(value);delete missing.provenance;
  assert(!validate(missing),'missing provenance accepted');
  const incompatible=structuredClone(value);incompatible.provenance.schema_version='unknown';
  assert(!validate(incompatible),'unknown schema accepted');
  const extra=structuredClone(value);extra.oracle_probability=.9;
  assert(!validate(extra),'hidden top-level field accepted');
  if (name==='DecisionResult') {
    const nested=structuredClone(value);nested.offers[0].support.hidden_probability=.8;
    assert(!validate(nested),'nested hidden field accepted');
    const badNumber=structuredClone(value);badNumber.offers[0].acceptance.mean=Infinity;
    assert(!validate(badNumber),'non-finite value accepted');
  }
  if (name==='EvaluationReport' || name==='ChallengerReport') {
    for (const field of ['cohort_n','advisory_supported_n','advisory_unsupported_n','missing_prediction_n','support_definition']) {
      const missingCount=structuredClone(value); delete missingCount.prediction[0][field];
      assert(!validate(missingCount),`absent ${field} accepted`);
      const invalidCount=structuredClone(value); invalidCount.prediction[0][field]=field==='support_definition'?'unknown':-1;
      assert(!validate(invalidCount),`invalid ${field} accepted`);
    }
  }
}
console.log(`${Object.keys(fixtures).length} actual response contracts passed; missing provenance, incompatible versions, hidden fields and non-finite values rejected.`);
