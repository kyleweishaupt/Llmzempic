// Checks for web/js/schema-model.js, run by tests/test_schema_model_js.py:
//   node schema_model_check.mjs <schema-model.js> <templates.json>
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const model = await import(pathToFileURL(process.argv[2]).href);
const templates = JSON.parse(readFileSync(process.argv[3], "utf8"));

const canon = (v) => Array.isArray(v) ? v.map(canon)
	: v && typeof v === "object" ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, canon(v[k])])) : v;
const roundTrip = (schema) => {
	const parsed = model.fromSchema(schema);
	assert.ok(parsed.ok, `not editable: ${parsed.reason}`);
	assert.deepEqual(canon(model.toSchema(parsed.root)), canon(schema));
	return parsed.root;
};

const SECTIONS = {
	type: "object",
	properties: {
		a: { type: "array", items: { type: "string", "x-dataset": "codes" }, minItems: 1, maxItems: 1 },
		b: { type: "array", items: { type: "string", "x-dataset": "codes", description: "Supporting" }, maxItems: 5 },
		c: { type: "string", "x-dataset": "codes" },
		t: { type: "array", items: { type: "string", enum: ["x", "y"] }, minItems: 1 },
	},
	required: ["a", "b"],
	additionalProperties: false,
};

for (const schema of [...templates, SECTIONS]) roundTrip(schema);

const root = roundTrip(SECTIONS);
const byName = Object.fromEntries(root.fields.map((f) => [f.name, f]));
assert.equal(byName.a.kind, "list");
assert.equal(byName.a.item.kind, "dataset");
assert.equal(byName.a.item.dataset, "codes");
assert.equal(byName.a.minItems, 1);
assert.equal(byName.a.maxItems, 1);
assert.equal(byName.c.kind, "dataset");
assert.equal(byName.t.kind, "tags");
assert.equal(byName.t.minItems, 1);

assert.equal(model.fromSchema({ type: "object", properties: { x: { anyOf: [] } } }).ok, false);

const fresh = model.newNode("dataset", "code");
const tree = { kind: "object", fields: [fresh], extra: {} };
assert.equal(model.problems(tree).errors.get(fresh.id), "Choose a data set.");
fresh.dataset = "codes";
assert.equal(model.problems(tree).errors.size, 0);
assert.deepEqual(model.toSchema({ ...tree, nullable: false, description: "" }).properties.code, { type: "string", "x-dataset": "codes" });

console.log("schema-model checks passed");
