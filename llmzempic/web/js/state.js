import { fromSchema } from "./schema-model.js";

// Shared UI state. Modules call each other through `app` (filled in by shell.js and main.js at boot)
// so views don't import one another in a cycle.
export const S = {
	system: null,
	catalog: null,
	projects: [],
	pid: null,
	view: null,
	config: null,
	examples: [],
	invalid: new Map(),
	badJson: new Map(),
	stats: null,
	savedChecks: "",
	tab: "task",

	page: 0,
	filter: "all",
	labelFilter: null,
	search: "",
	selection: new Set(),
	jsonEditing: new Set(),
	lastLabel: null,

	runs: [],
	runId: null,
	run: null,
	events: [],
	logOpen: false,
	mistakesOnly: false,
	tryInput: "",
	tryImage: null,
	snippets: null,
	snippetTab: "php",
	schemaMode: "visual",
};

export const app = {};

export const hardware = () => (S.system && S.system.hardware) || null;
export const activeJob = () => (S.system && S.system.active_job) || null;
export const isImageProject = () => Boolean(S.config && S.config.input_kind === "image");
export const catalogModel = (id) => (S.catalog ? S.catalog.models.find((m) => m.id === id) : null) || null;
export const modelsForKind = (kind) => (S.catalog ? S.catalog.models.filter((m) => Boolean(m.vision) === (kind === "image")) : []);

export function recommendedModel(kind) {
	const models = modelsForKind(kind);
	return (models.find((m) => m.recommended) || models[0] || {}).id || null;
}

export function exampleErrors(id) {
	if (S.badJson.has(id)) return [S.badJson.get(id)];
	return S.invalid.get(id) || [];
}

export const problemCount = () => S.examples.filter((e) => exampleErrors(e.id).length).length;

export function setExamplesData(data) {
	S.examples = data.examples;
	S.invalid = new Map(data.invalid.map((i) => [i.id, i.errors]));
	S.stats = data.stats;
	S.badJson = new Map();
}

export function applyChecks(data) {
	S.invalid = new Map(data.invalid.map((i) => [i.id, i.errors]));
	S.stats = data.stats;
}

export function duplicateCounts() {
	const groups = (S.stats && S.stats.duplicates) || [];
	const conflicts = groups.filter((g) => g.conflict).length;
	return { conflicts, plain: groups.length - conflicts, ids: new Set(groups.flatMap((g) => g.ids)) };
}

let treeCache = { schema: null, parsed: null };

// The output schema parsed for forms; cached until the schema object changes.
export function schemaTree() {
	if (treeCache.schema !== S.config.schema) treeCache = { schema: S.config.schema, parsed: fromSchema(S.config.schema) };
	return treeCache.parsed;
}
