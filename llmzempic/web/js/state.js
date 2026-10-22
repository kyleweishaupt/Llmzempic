import { fromSchema } from "./schema-model.js";

// Shared UI state. Modules call each other through `app` (filled in by shell.js and main.js at boot)
// so views don't import one another in a cycle.
export const S = {
	system: null,
	catalog: null,
	projects: [],
	projectsLoaded: false, // the sidebar's empty state waits for the first load
	pid: null,
	view: null,
	config: null,
	examples: [], // the current page only; the server holds the rest
	total: 0,
	counts: { all: 0, problems: 0, duplicates: 0 },
	checking: { running: false, done: 0, total: 0 },
	badJson: new Map(),
	stats: null,
	savedChecks: "",
	tab: "task",

	query: { filter: "all", search: "", label: null, offset: 0 },
	selection: new Set(),
	selectAll: false, // every example matching the query, not just the selected page rows
	busy: null, // what a running bulk change is doing, shown in the selection bar
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
	screen: "welcome",
	datasets: [],
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
	const row = S.examples.find((e) => e.id === id);
	return (row && row.errors) || [];
}

export const problemCount = () => S.counts.problems;

export function setExamplesData(data) {
	Object.assign(S, { examples: data.examples, total: data.total, counts: data.counts, checking: data.checking, badJson: new Map() });
	if (data.stats) S.stats = data.stats;
}

export function duplicateCounts() {
	const d = (S.stats && S.stats.duplicates) || { groups: 0, conflicts: 0 };
	return { conflicts: d.conflicts, plain: d.groups - d.conflicts };
}

let treeCache = { schema: null, parsed: null };

// The output schema parsed for forms; cached until the schema object changes.
export function schemaTree() {
	if (treeCache.schema !== S.config.schema) treeCache = { schema: S.config.schema, parsed: fromSchema(S.config.schema) };
	return treeCache.parsed;
}
