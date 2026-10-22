// JSON Schema <-> editable field tree. Pure: no DOM.
//
// Node: { id, kind, name?, required?, description, nullable, options?, minimum?, maximum?, minItems?, maxItems?,
//         dataset?, item?, fields?, extra }
// `extra` carries keys the editor doesn't model (title, format, additionalProperties, ...) so a
// round trip through the visual editor never drops them.

export const KINDS = [
	["text", "Text", "type"],
	["number", "Number", "hash"],
	["integer", "Whole number", "hash"],
	["boolean", "Yes / no", "toggle"],
	["choice", "One of a list", "radio"],
	["tags", "Several of a list", "tag"],
	["dataset", "Value from a data set", "database"],
	["list", "List", "list"],
	["object", "Group of fields", "braces"],
];

export const ITEM_KINDS = [["text", "Text"], ["number", "Number"], ["integer", "Whole number"], ["boolean", "Yes / no"], ["dataset", "Value from a data set"], ["object", "Group of fields"], ["any", "Anything"]];

export const DATASET_KEY = "x-dataset";

const BLOCKED = ["anyOf", "oneOf", "allOf", "not", "$ref", "$defs", "definitions", "if", "then", "else", "patternProperties",
	"prefixItems", "const", "dependentSchemas", "dependentRequired", "propertyNames", "unevaluatedProperties", "contains"];

const MODELED = {
	text: ["type", "description"],
	boolean: ["type", "description"],
	number: ["type", "description", "minimum", "maximum"],
	integer: ["type", "description", "minimum", "maximum"],
	choice: ["type", "description", "enum"],
	tags: ["type", "description", "items", "minItems", "maxItems"],
	list: ["type", "description", "items", "minItems", "maxItems"],
	dataset: ["type", "description", "x-dataset"],
	object: ["type", "description", "properties", "required"],
};

let seq = 0;
export const uid = () => `n${++seq}`;

class Unsupported extends Error {}

const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const typesOf = (schema) => (Array.isArray(schema.type) ? schema.type : schema.type === undefined ? [] : [schema.type]);
const where = (path) => (path ? `"${path}"` : "The top level");

function extras(schema, kind) {
	const keep = {};
	for (const [k, v] of Object.entries(schema)) if (!MODELED[kind].includes(k)) keep[k] = v;
	return keep;
}

function parseNode(schema, path) {
	if (!isObj(schema)) throw new Unsupported(`${where(path)} isn't a schema object.`);
	const blocked = Object.keys(schema).find((k) => BLOCKED.includes(k));
	if (blocked) throw new Unsupported(`${where(path)} uses "${blocked}".`);
	const types = typesOf(schema);
	const main = types.filter((t) => t !== "null");
	if (main.length > 1) throw new Unsupported(`${where(path)} allows several types.`);
	const node = { id: uid(), kind: null, description: typeof schema.description === "string" ? schema.description : "", nullable: types.includes("null") };
	let type = main[0];

	if (Array.isArray(schema.enum)) {
		const options = schema.enum.filter((v) => v !== null);
		if ((type && type !== "string") || !options.every((v) => typeof v === "string")) throw new Unsupported(`${where(path)} has options that aren't text.`);
		Object.assign(node, { kind: "choice", options, nullable: node.nullable || schema.enum.includes(null) });
	} else {
		if (!type && isObj(schema.properties)) type = "object";
		if (type === "string" && typeof schema[DATASET_KEY] === "string") Object.assign(node, { kind: "dataset", dataset: schema[DATASET_KEY] });
		else if (type === "string") node.kind = "text";
		else if (type === "boolean") node.kind = "boolean";
		else if (type === "number" || type === "integer") {
			Object.assign(node, { kind: type, minimum: isNum(schema.minimum) ? schema.minimum : null, maximum: isNum(schema.maximum) ? schema.maximum : null });
		} else if (type === "array") {
			const items = schema.items;
			Object.assign(node, { minItems: isNum(schema.minItems) ? schema.minItems : null, maxItems: isNum(schema.maxItems) ? schema.maxItems : null });
			if (isObj(items) && Array.isArray(items.enum) && items.enum.every((v) => typeof v === "string") && (!items.type || items.type === "string")) {
				const itemExtra = { ...items };
				delete itemExtra.type;
				delete itemExtra.enum;
				Object.assign(node, { kind: "tags", options: [...items.enum], itemExtra });
			} else {
				Object.assign(node, { kind: "list", item: items === undefined ? null : parseNode(items, `${path}[]`) });
			}
		} else if (type === "object") {
			const required = new Set(Array.isArray(schema.required) ? schema.required : []);
			const props = isObj(schema.properties) ? schema.properties : {};
			for (const name of required) if (!(name in props)) throw new Unsupported(`${where(path)} requires "${name}", which isn't defined.`);
			node.kind = "object";
			node.fields = Object.entries(props).map(([name, sub]) => ({
				...parseNode(sub, path ? `${path}.${name}` : name), name, required: required.has(name),
			}));
		} else {
			throw new Unsupported(`${where(path)} has no type.`);
		}
	}
	node.extra = extras(schema, node.kind);
	return node;
}

export function fromSchema(schema) {
	try {
		const root = parseNode(schema, "");
		if (root.kind !== "object" || root.nullable) return { ok: false, reason: "The top level isn't a group of fields." };
		return { ok: true, root };
	} catch (e) {
		if (e instanceof Unsupported) return { ok: false, reason: e.message };
		throw e;
	}
}

function counts(node, out) {
	if (isNum(node.minItems)) out.minItems = node.minItems;
	if (isNum(node.maxItems)) out.maxItems = node.maxItems;
}

function emit(node) {
	const t = (base) => (node.nullable ? [base, "null"] : base);
	const out = {};
	switch (node.kind) {
		case "text": out.type = t("string"); break;
		case "boolean": out.type = t("boolean"); break;
		case "number":
		case "integer":
			out.type = t(node.kind);
			if (isNum(node.minimum)) out.minimum = node.minimum;
			if (isNum(node.maximum)) out.maximum = node.maximum;
			break;
		case "choice":
			out.type = t("string");
			out.enum = node.nullable ? [...node.options, null] : [...node.options];
			break;
		case "dataset":
			out.type = t("string");
			out[DATASET_KEY] = node.dataset;
			break;
		case "tags":
			out.type = t("array");
			out.items = { type: "string", enum: [...node.options], ...(node.itemExtra || {}) };
			counts(node, out);
			break;
		case "list":
			out.type = t("array");
			if (node.item) out.items = emit(node.item);
			counts(node, out);
			break;
		case "object": {
			out.type = t("object");
			out.properties = Object.fromEntries(node.fields.map((f) => [f.name, emit(f)]));
			const required = node.fields.filter((f) => f.required).map((f) => f.name);
			if (required.length) out.required = required;
			break;
		}
		default:
			return { ...node.extra };
	}
	const typed = { type: out.type };
	if (node.description) typed.description = node.description;
	delete out.type;
	return { ...typed, ...out, ...node.extra };
}

export const toSchema = (root) => emit(root);

export function newNode(kind, name = "") {
	const node = { id: uid(), kind, name, required: true, description: "", nullable: false, extra: {} };
	if (kind === "choice" || kind === "tags") node.options = [];
	if (kind === "number" || kind === "integer") Object.assign(node, { minimum: null, maximum: null });
	if (kind === "list" || kind === "tags") Object.assign(node, { minItems: null, maxItems: null });
	if (kind === "list") node.item = newNode("text");
	if (kind === "dataset") node.dataset = "";
	if (kind === "object") Object.assign(node, { fields: [], extra: { additionalProperties: false } });
	return node;
}

// Change a field's kind, keeping what still makes sense (name, description, options).
export function changeKind(node, kind) {
	const next = newNode(kind, node.name);
	Object.assign(next, { id: node.id, required: node.required, description: node.description, nullable: node.nullable });
	if ((kind === "choice" || kind === "tags") && node.options) next.options = [...node.options];
	return next;
}

const NAME_OK = /^[A-Za-z_][A-Za-z0-9_]*$/;

// Blocking problems keep the schema from being saved; warnings are advice.
export function problems(root) {
	const out = { errors: new Map(), warnings: new Map() };
	const visit = (node) => {
		if (node.kind === "object") {
			const seen = new Map();
			for (const f of node.fields) {
				const name = f.name.trim();
				if (!name) out.errors.set(f.id, "Give this field a name.");
				else if (seen.has(name)) out.errors.set(f.id, `There's already a field named "${name}".`);
				else if (!NAME_OK.test(name)) out.warnings.set(f.id, "Use letters, digits and _ so code generators get clean names.");
				seen.set(name, f);
				visit(f);
			}
		}
		if ((node.kind === "choice" || node.kind === "tags") && !node.options.length) out.errors.set(node.id, "Add at least one option.");
		if (node.kind === "dataset" && !node.dataset) out.errors.set(node.id, "Choose a data set.");
		if (isNum(node.minItems) && isNum(node.maxItems) && node.minItems > node.maxItems) out.errors.set(node.id, "The minimum can't be more than the maximum.");
		if (node.kind === "list" && node.item) {
			visit(node.item);
			if (out.errors.has(node.item.id)) out.errors.set(node.id, out.errors.get(node.item.id));
		}
	};
	visit(root);
	return out;
}

// names: data set id -> display name, for the placeholder of data set fields.
export function sample(node, names = {}) {
	switch (node.kind) {
		case "text": return node.description ? `<${node.description.slice(0, 40)}>` : "text";
		case "number": return isNum(node.minimum) ? node.minimum : 0.5;
		case "integer": return isNum(node.minimum) ? node.minimum : 1;
		case "boolean": return true;
		case "choice": return node.options[0] ?? "";
		case "dataset": return `<value from ${names[node.dataset] || node.dataset || "a data set"}>`;
		case "tags": return node.options.slice(0, 2);
		case "list": return node.item ? [sample(node.item, names)] : [];
		case "object": return Object.fromEntries(node.fields.map((f) => [f.name, sample(f, names)]));
		default: return null;
	}
}

// Starting value for a new example: blanks for text and lists; labels stay unset until chosen.
export function blankValue(node) {
	switch (node.kind) {
		case "text": return "";
		case "list":
		case "tags": return [];
		case "object": {
			const out = {};
			for (const f of node.fields) {
				const v = blankValue(f);
				if (v !== undefined) out[f.name] = v;
			}
			return out;
		}
		default: return undefined;
	}
}

export function blankOutput(schema) {
	const parsed = fromSchema(schema);
	return parsed.ok ? blankValue(parsed.root) : {};
}

// Top-level fields with a fixed value set, in the order the dataset stats use.
export function labelFields(root) {
	return root.fields.filter((f) => f.kind === "boolean" || f.kind === "choice");
}

export function labelValues(node) {
	if (node.kind === "boolean") return node.nullable ? [true, false, null] : [true, false];
	return node.nullable ? [...node.options, null] : node.options;
}
