import { api } from "./api.js";
import { button, callout, h, icon, mount, pill } from "./dom.js";
import { lossChart } from "./chart.js";
import { toastError } from "./feedback.js";
import { duration, plural, when } from "./format.js";
import { evaluationView, metricTiles } from "./evaluation.js";
import { loadRuns, selectRun } from "./project.js";
import { monitorCard } from "./monitor.js";
import { tryPanel } from "./try.js";
import { S, app } from "./state.js";

const STATE = {
	completed: ["good", "Completed", "check"],
	failed: ["bad", "Failed", "x"],
	stopped: ["", "Stopped", "stop"],
	running: ["info", "Running", null],
	queued: ["info", "Queued", null],
};

export function statePill(status) {
	const state = (status && status.state) || "unknown";
	const [tone, text, iconName] = STATE[state] || ["", state, null];
	return pill(text, tone, { dot: !iconName, live: state === "running", iconName });
}

function planChips(st) {
	const p = st.plan;
	if (!p) return null;
	const chips = [p.device === "cuda" ? "GPU" : p.device.toUpperCase(), p.precision];
	if (p.four_bit) chips.push("4-bit QLoRA");
	if (p.unsloth) chips.push("Unsloth");
	return h("div", { class: "row" },
		chips.map((c) => h("span", { class: "chip" }, c)),
		st.train_seconds ? h("span", { class: "chip" }, icon("clock"), `Trained in ${duration(st.train_seconds)}`) : null,
	);
}

function progress(st) {
	const p = st.progress;
	if (!p || !p.total) return h("div", { class: "stack" }, h("div", { class: "progress indeterminate" }, h("span")), h("span", { class: "muted small" }, st.stage || "Starting"));
	const share = p.step / p.total;
	return h("div", { class: "stack run-progress" },
		h("div", { class: "spread" },
			h("span", { class: "num" }, h("b", {}, `${Math.round(share * 100)}%`), h("span", { class: "muted" }, `  Step ${p.step} of ${p.total}`)),
			p.eta ? h("span", { class: "muted" }, `About ${duration(p.eta)} left`) : null,
		),
		h("div", { class: "progress" }, h("span", { style: { width: `${share * 100}%` } })),
	);
}

// Training saves checkpoints as it goes, so a stopped or crashed run picks up where it left off.
async function resume(run, btn) {
	btn.disabled = true;
	try {
		await api(`/projects/${S.pid}/runs/${run.id}/resume`, { method: "POST" });
		await loadRuns();
		await selectRun(run.id);
		await app.refreshSystem();
		app.views.train();
		app.schedulePoll(500);
	} catch (e) {
		toastError(e);
		btn.disabled = false;
	}
}

function overlapNote(run) {
	if (!run.overlap) return null;
	return h("details", { class: "card card-pad skipped" },
		h("summary", {}, icon("alert"), `${plural(run.overlap, "held-out example")} nearly repeat a training example`),
		h("p", { class: "muted small" }, "Scores on near-copies of training inputs look better than the model is. Remove or reword one side of each pair."),
		h("div", { class: "row wrap" }, (run.overlap_pairs || []).map((p) => h("code", { class: "small", title: `${Math.round(p.similarity * 100)}% the same` }, `${p.held_out} ≈ ${p.train}`))),
	);
}

function skippedNote(run) {
	const s = run.skipped_examples;
	if (!run.skipped || !s) return null;
	return h("details", { class: "card card-pad skipped" },
		h("summary", {}, icon("alert"), `${plural(run.skipped, "example")} skipped: longer than ${s.max_length.toLocaleString()} tokens`),
		h("p", { class: "muted small" }, "Raise Max tokens per example on the Model tab to train on them, or shorten their inputs."),
		h("div", { class: "row wrap" }, s.ids.map((id) => h("code", { class: "small" }, id))),
		run.skipped > s.ids.length ? h("p", { class: "faint small" }, `…and ${(run.skipped - s.ids.length).toLocaleString()} more (listed in skipped.json in the run folder).`) : null,
	);
}

export function renderRunDetail(el) {
	const run = S.run;
	if (!run) { mount(el); return; }
	const st = run.status || {};
	const running = st.state === "running" || st.state === "queued";
	const chart = lossChart(S.events);
	const log = h("pre", { class: "code-block log nowrap" }, run.log || "(empty)");
	const logBox = h("details", { class: "card log-card", open: S.logOpen || running },
		h("summary", {}, icon("terminal"), "Training log"), log);
	logBox.addEventListener("toggle", () => { S.logOpen = logBox.open; if (logBox.open) log.scrollTop = log.scrollHeight; });
	mount(el,
		h("div", { class: "run-head" },
			h("div", { class: "stack" },
				h("div", { class: "row" }, h("h2", {}, `Run ${when(run.created) || run.id}`), statePill(st)),
				h("span", { class: "muted small" }, run.base_model, running && st.stage ? `, ${st.stage.toLowerCase()}` : ""),
			),
			running
				? button("Stop", { icon: "stop", kind: "secondary", onclick: () => app.stopRun(run.id) })
				: h("div", { class: "row" },
					run.resumable ? button("Resume", { icon: "play", kind: "secondary", title: "Continue from the last checkpoint", onclick: (e) => resume(run, e.currentTarget) }) : null,
					button(null, { icon: "trash", kind: "ghost", title: "Delete this run", onclick: () => app.deleteRun(run.id) })),
		),
		planChips(st),
		running ? progress(st) : null,
		st.error ? callout("bad", "alert-circle", h("b", {}, "Training failed. "), st.error) : null,
		skippedNote(run),
		overlapNote(run),
		chart ? h("section", { class: "card card-pad" }, chart) : null,
		run.evaluation ? evaluationView(run.evaluation) : run.metrics ? metricTiles(run.metrics) : null,
		st.state === "completed" && run.monitored ? monitorCard(run) : null,
		st.state === "completed" ? tryPanel(run) : null,
		st.state === "completed" ? h("div", { class: "export-cta" },
			h("div", {}, h("b", {}, "Happy with the results?"), h("p", { class: "muted small" }, "Export a GGUF model and client code for your apps.")),
			button("Export this model", { icon: "box", kind: "primary", onclick: () => app.showTab("export") })) : null,
		logBox,
	);
	log.scrollTop = log.scrollHeight;
}
