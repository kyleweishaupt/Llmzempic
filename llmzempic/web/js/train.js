import { api } from "./api.js";
import { $, button, emptyState, h, icon, mount } from "./dom.js";
import { confirmAction, saver, toast, toastError } from "./feedback.js";
import { labelText, pct, plural, shortModel, when } from "./format.js";
import { loadRuns, selectRun } from "./project.js";
import { renderRunDetail, statePill } from "./run-view.js";
import { S, activeJob, app, catalogModel, duplicateCounts, hardware, isImageProject, problemCount } from "./state.js";

function readiness() {
	const hw = hardware();
	const bad = problemCount();
	const n = S.examples.length;
	const entry = catalogModel(S.config.base_model);
	const fits = !entry || Boolean(entry.vision) === isImageProject();
	const checks = [
		[S.view.schema_errors.length === 0, "Output schema is valid", "Fix the output schema on the Task tab", "task"],
		[n >= 2 && bad === 0, `${plural(n, "example")}, all valid`, n < 2 ? "Add at least 2 examples" : `Fix ${plural(bad, "example")} on the Examples tab`, "examples"],
		[Boolean(S.config.base_model) && fits, `Base model: ${entry ? entry.name : shortModel(S.config.base_model)}`,
			!S.config.base_model ? "Choose a base model" : isImageProject() ? "Choose a vision model" : "Choose a text model", "model"],
		[Boolean(hw && hw.can_train), "Training libraries installed",
			`Install the training libraries (${hw ? hw.missing_train_packages.join(", ") || "PyTorch" : "…"}) with the setup script`, null],
	];
	if (isImageProject() && hw && hw.can_train) {
		checks.push([hw.can_train_images !== false, "Image libraries installed",
			`Install ${(hw.missing_image_packages || []).join(" and ")} with the setup script`, null]);
	}
	const warnings = [];
	if (n >= 2 && n < 20) warnings.push("Fewer than 20 examples: results will be unreliable. 100+ is better.");
	if (n >= 2 && n < 10) warnings.push("With fewer than 10 examples there's no held-out set, so scores use training examples.");
	if (S.stats) {
		for (const f of S.stats.fields) {
			const unseen = f.values.filter((v) => v.count === 0).map((v) => labelText(v.value));
			if (n >= 2 && unseen.length) warnings.push(`${f.field}: no examples of ${unseen.join(", ")}. The model can't learn a value it never sees.`);
		}
		const { conflicts } = duplicateCounts();
		if (conflicts) warnings.push(`${plural(conflicts, "input")} appear more than once with different answers; the model can't learn both.`);
	}
	if (hw && hw.device !== "cuda") warnings.push("No GPU: training is slow for models above about 0.5B parameters.");
	return { ok: checks.every((c) => c[0]), checks, warnings };
}

async function startTraining(btn) {
	btn.disabled = true;
	try {
		await saver.flush();
		const run = await api(`/projects/${S.pid}/runs`, { method: "POST" });
		await loadRuns();
		await selectRun(run.id);
		await app.refreshSystem();
		renderTrain();
		toast("Training started");
		app.schedulePoll(500);
	} catch (e) {
		toastError(e);
		btn.disabled = false;
	}
}

function readyCard() {
	const r = readiness();
	const job = activeJob();
	const start = button("Start training", { icon: "play", kind: "primary", size: "lg", disabled: !r.ok || Boolean(job), cls: "start-btn" });
	start.addEventListener("click", () => startTraining(start));
	return h("section", { class: "card card-pad stack ready" },
		h("h2", { class: "section-title" }, r.ok ? "Ready to train" : "Before you train"),
		h("ul", { class: "checklist" }, r.checks.map(([ok, good, todo, tab]) => h("li", { class: ok ? "ok" : "todo" },
			icon(ok ? "check-circle" : "x-circle"),
			!ok && tab ? h("button", { type: "button", class: "link-btn", onclick: () => app.showTab(tab) }, todo) : h("span", {}, ok ? good : todo)))),
		r.warnings.length ? h("ul", { class: "warnings" }, r.warnings.map((w) => h("li", {}, icon("alert"), h("span", {}, w)))) : null,
		start,
		job ? h("p", { class: "faint small" }, job.kind === "train" && job.project_id === S.pid
			? "Training is in progress."
			: `Waiting for the running ${job.kind} job to finish.`) : null,
	);
}

function runItem(run) {
	const m = run.metrics;
	return h("button", {
		type: "button", class: "run-item", "aria-current": run.id === S.runId ? "true" : null,
		onclick: async () => {
			try { await selectRun(run.id); } catch (e) { toastError(e); return; }
			renderTrain();
		},
	},
		h("span", { class: "stack run-item-text" },
			h("span", { class: "row-tight" }, h("b", {}, when(run.created) || run.id)),
			h("span", { class: "faint small ellipsis" }, `${shortModel(run.base_model)}, ${run.train_count} train / ${run.eval_count} held out`),
		),
		h("span", { class: "run-item-score" },
			m && m.exact_match !== undefined ? h("span", { class: "num score" }, pct(m.exact_match)) : null,
			statePill(run.status),
		),
	);
}

export async function deleteRun(runId) {
	const ok = await confirmAction({ title: "Delete this run?", text: "Its trained adapter, scores and exports are removed from disk.", action: "Delete run", danger: true });
	if (!ok) return;
	try {
		await api(`/projects/${S.pid}/runs/${runId}`, { method: "DELETE" });
		if (S.runId === runId) Object.assign(S, { runId: null, run: null, events: [] });
		await loadRuns();
		if (!S.runId && S.runs.length) await selectRun(S.runs[0].id);
		renderTrain();
	} catch (e) {
		toastError(e);
	}
}

export function renderTrain() {
	const detail = h("div", { class: "run-detail stack-lg" });
	mount($("#tab-train"), h("div", { class: "train-grid" },
		h("div", { class: "train-side stack-lg" },
			readyCard(),
			S.runs.length ? h("section", { class: "stack" },
				h("h3", { class: "section-title" }, "Runs", h("span", { class: "faint" }, String(S.runs.length))),
				h("div", { class: "run-list" }, S.runs.map(runItem)),
			) : null,
		),
		S.run ? detail : emptyState("flask", "No runs yet", "When your examples are ready, start training. You'll see the loss fall live, then scores on examples the model never saw."),
	));
	if (S.run) renderRunDetail(detail);
}

app.deleteRun = deleteRun;
app.renderRunDetail = () => {
	const el = $("#tab-train .run-detail");
	if (el) renderRunDetail(el);
	for (const item of document.querySelectorAll("#tab-train .run-list")) mount(item, S.runs.map(runItem));
};
