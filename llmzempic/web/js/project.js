import { api } from "./api.js";
import { saver, toastError } from "./feedback.js";
import { S, app, setExamplesData } from "./state.js";

export async function loadProjects() {
	S.projects = (await api("/projects")).projects;
	S.projectsLoaded = true;
}

export async function loadRuns() {
	S.runs = (await api(`/projects/${S.pid}/runs`)).runs;
}

export async function selectRun(runId) {
	const data = await api(`/projects/${S.pid}/runs/${runId}`);
	S.runId = runId;
	S.run = data;
	S.events = data.events;
	S.snippets = null;
}

export async function pollRun() {
	const data = await api(`/projects/${S.pid}/runs/${S.runId}?since=${S.events.length}&items=false`);
	S.events.push(...data.events);
	data.evaluation = S.run && S.run.evaluation;
	S.run = data;
	const i = S.runs.findIndex((r) => r.id === S.runId);
	if (i >= 0) S.runs[i] = { ...S.runs[i], status: data.status, export: data.export, metrics: data.metrics };
}

// Example validity depends on the schema and on the input kind (image projects require an image).
const checksKey = (config) => `${config.input_kind}\n${JSON.stringify(config.schema)}`;

export const pageSize = () => (S.config && S.config.input_kind === "image" ? 48 : 20);

export function examplesUrl({ stats = true, limit } = {}) {
	const q = S.query;
	const params = new URLSearchParams({ filter: q.filter, offset: q.offset, limit: limit ?? pageSize(), stats: stats ? 1 : 0 });
	if (q.search) params.set("search", q.search);
	if (q.label) {
		params.set("field", q.label.field);
		params.set("value", JSON.stringify(q.label.value));
		params.set("kind", q.label.kind);
	}
	return `/projects/${S.pid}/examples?${params}`;
}

export async function loadPage({ stats = true } = {}) {
	setExamplesData(await api(examplesUrl({ stats })));
	watchChecks();
}

// Counts, stats and checking progress, plus fresh errors for the rows on screen, without replacing
// rows: someone may be typing in one.
export async function refreshChecks({ stats = true } = {}) {
	const pid = S.pid;
	const wasChecking = S.checking.running;
	const data = await api(examplesUrl({ stats }));
	if (S.pid !== pid) return;
	Object.assign(S, { total: data.total, counts: data.counts, checking: data.checking, stats: data.stats || S.stats });
	for (const row of data.examples) {
		const local = S.examples.find((e) => e.id === row.id);
		if (local && !saver.timers[`ex:${row.id}`]) local.errors = row.errors;
	}
	app.examplesChecked && app.examplesChecked();
	if (S.tab === "train" && wasChecking !== S.checking.running) app.views.train();
	app.renderStepper();
	watchChecks();
}

// While the server checks examples (after an import or a schema change), keep counts current on every
// tab. Checking doesn't change the stats, so polls skip them.
let watchTimer = null;
function watchChecks() {
	clearTimeout(watchTimer);
	if (S.checking.running) watchTimer = setTimeout(() => refreshChecks({ stats: false }).catch(toastError), 1000);
}

export async function loadProject(pid) {
	const [view, runs] = await Promise.all([api(`/projects/${pid}`), api(`/projects/${pid}/runs`)]);
	Object.assign(S, {
		pid, view, config: view.config, savedChecks: checksKey(view.config), runs: runs.runs,
		query: { filter: "all", search: "", label: null, offset: 0 }, selectAll: false,
		runId: null, run: null, events: [], snippets: null, tryInput: "", tryImage: null, importWarnings: null, stats: null,
	});
	S.selection.clear();
	S.jsonEditing.clear();
	// Statistics read every example; the first page shouldn't wait for them.
	await loadPage({ stats: false });
	refreshChecks().catch(toastError);
	if (S.runs.length) await selectRun(S.runs[0].id);
}

export function setSaveState(text) {
	const el = document.getElementById("save-state");
	if (el) el.textContent = text;
}

export function scheduleConfigSave() {
	const pid = S.pid;
	const config = structuredClone(S.config);
	setSaveState("Saving…");
	saver.schedule("config", async () => {
		try {
			const view = await api(`/projects/${pid}`, { method: "PUT", body: { config } });
			if (S.pid !== pid) return;
			S.view = view;
			setSaveState("Saved");
			app.configSaved && app.configSaved();
			const checks = checksKey(config);
			if (checks !== S.savedChecks) {
				S.savedChecks = checks;
				refreshChecks().catch(toastError);
			}
			const p = S.projects.find((x) => x.id === pid);
			if (p && (p.name !== config.name || p.base_model !== config.base_model || p.input_kind !== config.input_kind)) {
				Object.assign(p, { name: config.name, base_model: config.base_model, input_kind: config.input_kind });
				app.renderSidebar();
			}
			app.renderStepper();
		} catch (e) {
			setSaveState("Not saved");
			toastError(e);
		}
	});
}

// One example's edits, saved shortly after typing stops and checked on the server right away.
export function saveExample(ex) {
	const pid = S.pid;
	setSaveState("Saving…");
	saver.schedule(`ex:${ex.id}`, async () => {
		try {
			const row = await api(`/projects/${pid}/examples/${encodeURIComponent(ex.id)}`, {
				method: "PUT", body: { input: ex.input, output: ex.output, ...(ex.image ? { image: ex.image } : {}) },
			});
			if (S.pid !== pid) return;
			ex.errors = row.errors;
			setSaveState("Saved");
			await refreshChecks();
		} catch (e) {
			setSaveState("Not saved");
			toastError(e);
		}
	});
}
