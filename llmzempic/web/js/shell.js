import { api } from "./api.js";
import { $, $$, button, h, icon, menu, mount, pill, seg } from "./dom.js";
import { confirmAction, saver, toastError } from "./feedback.js";
import { duration, pct, plural, shortModel } from "./format.js";
import { renderDatasets } from "./datasets.js";
import { duplicateProject, exportPackageDialog, importPackage } from "./packages.js";
import { loadProject, loadProjects, selectRun } from "./project.js";
import { S, activeJob, app, catalogModel, hardware, isImageProject, problemCount } from "./state.js";
import { applyTheme, currentTheme } from "./theme.js";

const STEPS = [["task", "Task"], ["examples", "Examples"], ["model", "Model"], ["train", "Train"], ["export", "Export"]];
const JOB_LABEL = { train: "Training", export: "Exporting", predict: "Trying a model" };

// ---------- sidebar ----------

function projectItem(p) {
	const last = p.last_run && p.last_run.status;
	const state = last && last.state;
	const tone = state === "completed" ? "good" : state === "failed" ? "bad" : state === "running" || state === "queued" ? "info" : "";
	return h("button", {
		type: "button", class: "project-item", "aria-current": p.id === S.pid ? "page" : null,
		onclick: () => { closeDrawer(); openProject(p.id); },
	},
		h("span", { class: "project-icon" }, icon(p.input_kind === "image" ? "image" : "text")),
		h("span", { class: "grow" },
			h("span", { class: "project-name ellipsis" }, p.name),
			h("span", { class: "project-sub ellipsis" }, shortModel(p.base_model) || "No model chosen"),
		),
		state ? h("span", { class: `status-dot ${tone}`, title: `Last run ${state}` }) : null,
	);
}

function jobCard() {
	const job = activeJob();
	if (!job) return null;
	const project = S.projects.find((p) => p.id === job.project_id);
	const progress = S.run && S.run.id === job.run_id && job.kind === "train" ? S.run.status.progress : null;
	const share = progress && progress.total ? progress.step / progress.total : null;
	return h("div", { class: "job-card" },
		h("div", { class: "spread" },
			h("span", { class: "row-tight" }, h("span", { class: "spinner" }), h("b", {}, JOB_LABEL[job.kind] || job.kind)),
			h("span", { class: "faint num" }, duration(Date.now() / 1000 - job.started)),
		),
		h("div", { class: "muted ellipsis" }, project ? project.name : job.project_id),
		h("div", { class: share === null ? "progress indeterminate" : "progress" }, h("span", { style: share === null ? null : { width: `${Math.round(share * 100)}%` } })),
		button("View", { kind: "ghost", size: "sm", onclick: () => { closeDrawer(); openProject(job.project_id, job.kind === "export" ? "export" : "train", job.run_id); } }),
	);
}

function hardwareLine() {
	const hw = hardware();
	if (!hw) return h("div", { class: "hw faint" }, h("span", { class: "spinner" }), "Detecting hardware…");
	const gpu = hw.device === "cuda" ? `${hw.gpus[0].name.replace(/^NVIDIA (GeForce )?/, "")}, ${hw.vram_gb} GB` : hw.device === "mps" ? "Apple GPU" : "CPU only";
	return h("div", { class: hw.can_train ? "hw" : "hw bad", title: hw.can_train ? `${hw.ram_gb || "?"} GB RAM` : `Missing: ${hw.missing_train_packages.join(", ")}` },
		icon("cpu"), h("span", { class: "ellipsis" }, hw.can_train ? gpu : "Training libraries missing"));
}

export function renderSidebar() {
	$("#datasets-link").setAttribute("aria-current", S.screen === "datasets" ? "page" : "false");
	const list = $("#project-list");
	mount(list, S.projects.length
		? S.projects.map(projectItem)
		: h("p", { class: "side-empty" }, S.projectsLoaded ? "No projects yet." : "Loading projects…"));
	mount($("#job-slot"), jobCard());
	mount($("#hw-slot"), hardwareLine());
	mount($("#theme-slot"), seg([["system", null, "monitor"], ["light", null, "sun"], ["dark", null, "moon"]], currentTheme(),
		(t) => { applyTheme(t); renderSidebar(); }, { label: "Theme" }));
	for (const [i, b] of $$("#theme-slot button").entries()) b.title = ["Match the system", "Light", "Dark"][i];
}

export function openDrawer() { document.body.classList.add("drawer-open"); }
export function closeDrawer() { document.body.classList.remove("drawer-open"); }

// ---------- project header & stepper ----------

function stepInfo(tab) {
	const job = activeJob();
	const mine = job && job.project_id === S.pid;
	if (tab === "task") {
		const errors = (S.view.schema_errors || []).length;
		const fields = Object.keys((S.config.schema && S.config.schema.properties) || {}).length;
		if (errors) return ["problem", "Schema has problems"];
		return [S.config.instructions.trim() && fields ? "done" : "todo", plural(fields, "output field")];
	}
	if (tab === "examples") {
		const n = S.counts.all;
		const bad = problemCount();
		if (S.checking.running) return ["running", `Checking ${S.checking.done.toLocaleString()} of ${S.checking.total.toLocaleString()}`];
		if (bad) return ["problem", `${bad.toLocaleString()} to fix`];
		return [n >= 2 ? "done" : "todo", n ? plural(n, "example") : "None yet"];
	}
	if (tab === "model") {
		const entry = catalogModel(S.config.base_model);
		if (entry && Boolean(entry.vision) !== isImageProject()) return ["problem", isImageProject() ? "Can't read images" : "Vision model"];
		return [S.config.base_model ? "done" : "todo", entry ? entry.name : shortModel(S.config.base_model) || "Not chosen"];
	}
	if (tab === "train") {
		if (mine && job.kind === "train") return ["running", "Training…"];
		const done = S.runs.filter((r) => r.status.state === "completed");
		if (!done.length) return ["todo", S.runs.length ? plural(S.runs.length, "run") : "Not trained"];
		const best = Math.max(...done.map((r) => (r.metrics && r.metrics.exact_match) || 0));
		return ["done", `Best exact match ${pct(best)}`];
	}
	if (mine && job.kind === "export") return ["running", "Exporting…"];
	return S.runs.some((r) => r.export && r.export.state === "completed") ? ["done", "Exported"] : ["todo", "Not exported"];
}

export function renderStepper() {
	if (!S.pid) return;
	mount($("#stepper"), STEPS.map(([tab, label], i) => {
		const [state, caption] = stepInfo(tab);
		const mark = state === "done" ? icon("check") : state === "problem" ? icon("alert") : state === "running" ? h("span", { class: "spinner" }) : String(i + 1);
		return h("button", {
			type: "button", class: `step ${state}`, "aria-current": S.tab === tab ? "step" : null,
			onclick: () => showTab(tab),
		},
			h("span", { class: "step-mark" }, mark),
			h("span", { class: "step-text" }, h("span", { class: "step-label" }, label), h("span", { class: "step-caption ellipsis" }, caption)),
		);
	}));
}

export function renderProjectHead() {
	mount($("#project-title"), S.config.name || S.pid);
	mount($("#kind-slot"), pill(isImageProject() ? "Images" : "Text", "", { iconName: isImageProject() ? "image" : "text" }));
	mount($("#project-menu"), menu(button(null, { icon: "more", kind: "ghost", title: "Project actions" }), [
		{ label: "Export package…", icon: "download", onclick: exportPackageDialog },
		{ label: "Duplicate project", icon: "copy", onclick: duplicateProject },
		"-",
		{ label: "Delete project", icon: "trash", danger: true, onclick: deleteProject },
	]));
}

async function deleteProject() {
	const ok = await confirmAction({
		title: `Delete "${S.config.name}"?`,
		text: "Its examples, images, trained runs and exports are removed from disk. This can't be undone.",
		action: "Delete project", danger: true,
	});
	if (!ok) return;
	try {
		await api(`/projects/${S.pid}`, { method: "DELETE" });
		S.pid = null;
		history.replaceState(null, "", "#");
		await loadProjects();
		renderSidebar();
		renderWelcome();
	} catch (e) {
		toastError(e);
	}
}

// ---------- navigation ----------

export function showTab(tab) {
	if (!STEPS.some(([t]) => t === tab)) tab = "task";
	if (tab !== S.tab) window.scrollTo(0, 0);
	S.tab = tab;
	history.replaceState(null, "", `#${S.pid}/${tab}`);
	for (const pane of $$(".tab-pane")) pane.hidden = pane.id !== `tab-${tab}`;
	app.views[tab]();
	renderStepper();
}

export async function openProject(pid, tab, runId) {
	await saver.flush();
	try {
		if (pid !== S.pid) await loadProject(pid);
		if (runId && runId !== S.runId) await selectRun(runId);
	} catch (e) {
		toastError(e);
		return;
	}
	S.screen = "project";
	$("#welcome").hidden = true;
	$("#datasets-page").hidden = true;
	$("#project").hidden = false;
	renderProjectHead();
	renderSidebar();
	showTab(tab || S.tab);
}

export async function openDatasets() {
	await saver.flush();
	Object.assign(S, { screen: "datasets", pid: null });
	history.replaceState(null, "", "#datasets");
	$("#welcome").hidden = true;
	$("#project").hidden = true;
	$("#datasets-page").hidden = false;
	renderSidebar();
	window.scrollTo(0, 0);
	renderDatasets();
}

export function renderWelcome() {
	S.screen = "welcome";
	$("#project").hidden = true;
	$("#datasets-page").hidden = true;
	const el = $("#welcome");
	el.hidden = false;
	const steps = [
		["Task", "Say what the model should do and design the JSON it answers with."],
		["Examples", "Add inputs, text or images, with the exact answer you expect."],
		["Model", "Pick an open model that fits your GPU."],
		["Train", "Fine-tune with LoRA, then score it on examples it never saw."],
		["Export", "Get a GGUF model plus PHP, C# and Python clients."],
	];
	mount(el,
		h("div", { class: "welcome-hero" },
			h("h1", {}, "Teach a small model to answer in your JSON"),
			h("p", { class: "lede" }, "Describe a task, show it examples, and Llmzempic fine-tunes an open model on your machine into a specialist that returns the structure your code expects."),
			h("div", { class: "row" },
				button("New project", { kind: "primary", size: "lg", icon: "plus", onclick: () => app.newProject() }),
				button("Import a package", { kind: "secondary", size: "lg", icon: "upload", onclick: importPackage })),
		),
		h("ol", { class: "welcome-steps" }, steps.map(([title, text], i) => h("li", {},
			h("span", { class: "step-mark" }, String(i + 1)), h("b", {}, title), h("span", { class: "muted" }, text)))),
		S.projects.length ? h("section", { class: "stack" },
			h("h2", { class: "section-title" }, "Your projects"),
			h("div", { class: "project-cards" }, S.projects.map((p) => h("button", { type: "button", class: "project-card card", onclick: () => openProject(p.id) },
				h("span", { class: "project-icon" }, icon(p.input_kind === "image" ? "image" : "text")),
				h("b", { class: "ellipsis" }, p.name),
				h("span", { class: "faint small ellipsis" }, shortModel(p.base_model)),
			))),
		) : null,
	);
}
