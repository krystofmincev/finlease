"""Generate browser contracts from the actual FastAPI OpenAPI components."""

import json

from lease_lab.api.app import app
from lease_lab.config import ROOT

schema = app.openapi()
target = ROOT / "ui/src/api"
target.mkdir(parents=True, exist_ok=True)
(target / "openapi.json").write_text(json.dumps(schema, indent=2) + "\n")
# OpenAPI 3.1 components are JSON Schema. Relocate only their references for Ajv.
components = json.loads(
    json.dumps(schema["components"]["schemas"]).replace(
        "#/components/schemas/", "#/$defs/"
    )
)
(target / "schemas.json").write_text(
    json.dumps(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": components,
        },
        indent=2,
    )
    + "\n"
)
print("Exported actual OpenAPI and JSON Schema components.")
