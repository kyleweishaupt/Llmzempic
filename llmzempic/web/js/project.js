import { api } from "./api.js";
import { saver, toastError } from "./feedback.js";
import { S, app, applyChecks, setExamplesData } from "./state.js";

export async function loadProjects() {
	S.projects = (await api("/projects")).projects;
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

export async function loadProject(pid) {
	const [view, examples, runs] = await Promise.all([
		api(`/projects/${pid}`), api(`/projects/${pid}/examples`), api(`/projects/${pid}/runs`),
	]);
	Object.assign(S, {
		pid, view, config: view.config, savedChecks: checksKey(view.config), runs: runs.runs,
		page: 0, filter: "all", labelFilter: null, search: "", runId: null, run: null, events: [],
		snippets: null, tryInput: "", tryImage: null, importWarnings: [],
	});
	S.selection.clear();
	S.jsonEditing.clear();
	setExamplesData(examples);
	if (S.runs.length) await selectRun(S.runs[0].id);
}

export function setSaveState(text) {
	const el = document.getElementById("save-state");
	if (el) el.textContent = text;
}

// Schema checks and label counts depend on the schema; refresh them without touching unsaved example edits.
export async function refreshExampleChecks(pid) {
	const data = await api(`/projects/${pid}/examples`);
	if (S.pid !== pid) return;
	applyChecks(data);
	app.examplesChecked && app.examplesChecked();
	app.renderStepper();
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
				refreshExampleChecks(pid).catch(toastError);
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

export function scheduleExamplesSave() {
	const pid = S.pid;
	const examples = S.examples;
	setSaveState("Saving…");
	saver.schedule("examples", async () => {
		try {
			const data = await api(`/projects/${pid}/examples`, { method: "PUT", body: { examples } });
			if (S.pid !== pid) return;
			applyChecks(data);
			setSaveState("Saved");
			app.examplesChecked && app.examplesChecked();
			app.renderStepper();
		} catch (e) {
			setSaveState("Not saved");
			toastError(e);
		}
	});
}
