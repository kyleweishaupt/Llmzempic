import { api } from "./api.js";
import { $ } from "./dom.js";
import { newProjectDialog, settingsDialog } from "./dialogs.js";
import { renderExamples } from "./examples.js";
import { renderExport } from "./export.js";
import { saver, toastError } from "./feedback.js";
import { renderModel } from "./model.js";
import { loadProjects, loadRuns, pollRun, selectRun } from "./project.js";
import { closeDrawer, openDrawer, openProject, renderProjectHead, renderSidebar, renderStepper, renderWelcome, showTab } from "./shell.js";
import { S, app } from "./state.js";
import { renderTask } from "./task.js";
import { renderTrain } from "./train.js";

Object.assign(app, {
	views: { task: renderTask, examples: renderExamples, model: renderModel, train: renderTrain, export: renderExport },
	showTab, openProject, renderSidebar, renderStepper, renderProjectHead,
	newProject: newProjectDialog,
	refreshSystem: async () => {
		S.system = await api("/system");
		renderSidebar();
	},
	schedulePoll,
	stopRun,
});

async function stopRun(runId) {
	try {
		await api(`/projects/${S.pid}/runs/${runId}/stop`, { method: "POST" });
		await refreshAfterJob();
	} catch (e) {
		toastError(e);
	}
}

// ---------- polling: one job at a time, so /system says whether anything is running ----------

let pollTimer = null;
function schedulePoll(ms) {
	clearTimeout(pollTimer);
	pollTimer = setTimeout(poll, ms);
}

async function refreshAfterJob() {
	S.system = await api("/system");
	await loadProjects();
	renderSidebar();
	if (!S.pid) return;
	await loadRuns();
	if (S.runId) await selectRun(S.runId);
	if (S.tab === "train" || S.tab === "export") showTab(S.tab);
	else renderStepper();
}

async function poll() {
	let next = 5000;
	try {
		const before = S.system && S.system.active_job;
		S.system = await api("/system");
		const job = S.system.active_job;
		if (before && !job) {
			await refreshAfterJob();
		} else if (job && S.pid === job.project_id) {
			next = 2000;
			if (job.run_id !== S.runId && !before) await selectRun(job.run_id);
			if (job.run_id === S.runId) {
				await pollRun();
				if (S.tab === "train") app.renderRunDetail();
				if (S.tab === "export") renderExport();
			}
			renderStepper();
			renderSidebar();
		} else {
			if (job) next = 3000;
			renderSidebar();
		}
	} catch {
		next = 5000;
	}
	schedulePoll(next);
}

// ---------- boot ----------

function bindChrome() {
	$("#new-project-btn").addEventListener("click", () => { closeDrawer(); newProjectDialog(); });
	$("#settings-btn").addEventListener("click", () => { closeDrawer(); settingsDialog(); });
	$("#drawer-btn").addEventListener("click", openDrawer);
	$("#scrim").addEventListener("click", closeDrawer);
	document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
	// The app only rewrites the hash with replaceState, so this fires for typed URLs and back/forward.
	window.addEventListener("hashchange", () => {
		const [pid, tab] = decodeURIComponent(location.hash.slice(1)).split("/");
		if (pid && S.projects.some((p) => p.id === pid)) openProject(pid, tab || "task");
	});
	window.addEventListener("beforeunload", (e) => {
		if (!saver.pending()) return;
		saver.flush();
		e.preventDefault();
	});
}

async function boot() {
	bindChrome();
	renderSidebar();
	try {
		[S.system, S.catalog] = await Promise.all([api("/system"), api("/catalog")]);
		await loadProjects();
		renderSidebar();
		const [pid, tab] = decodeURIComponent(location.hash.slice(1)).split("/");
		if (pid && S.projects.some((p) => p.id === pid)) await openProject(pid, tab || "task");
		else renderWelcome();
	} catch (e) {
		renderWelcome();
		toastError(`Couldn't reach the Llmzempic server: ${e.message}`);
	}
	schedulePoll(2000);
}

boot();
