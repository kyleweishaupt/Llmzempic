import { api } from "./api.js";
import { $, button, callout, emptyState, field, h, mount } from "./dom.js";
import { toast, toastError } from "./feedback.js";
import { bytes, pct, shortModel, when } from "./format.js";
import { loadRuns, selectRun } from "./project.js";
import { statePill } from "./run-view.js";
import { S, activeJob, app, hardware } from "./state.js";

const GGUF = {
	q8_0: ["Q8_0", "Near-lossless, about 1 byte per parameter. Recommended."],
	q5_k_m: ["Q5_K_M", "Smaller, very close to Q8 quality."],
	q4_k_m: ["Q4_K_M", "Smallest and fastest on CPU."],
	f16: ["F16", "Full precision, largest file."],
	bf16: ["BF16", "Full precision, largest file."],
	none: ["No GGUF", "Adapter and client files only."],
};

const SNIPPETS = [["php", "PHP"], ["csharp", "C#"], ["python", "Python"], ["llama_server", "llama-server"]];

async function startExport(opts, btn) {
	btn.disabled = true;
	try {
		await api(`/projects/${S.pid}/runs/${S.runId}/export`, { method: "POST", body: opts });
		await app.refreshSystem();
		await selectRun(S.runId);
		await loadRuns();
		renderExport();
		app.schedulePoll(500);
	} catch (e) {
		toastError(e);
		btn.disabled = false;
	}
}

function optionsCard(run) {
	const job = activeJob();
	const hw = hardware();
	const formats = S.catalog.gguf_types;
	let gguf = "q8_0";
	const formatList = h("div", { class: "format-list", role: "radiogroup", "aria-label": "Model file format" });
	const drawFormats = () => mount(formatList, formats.map((t) => {
		const [name, text] = GGUF[t] || [t, ""];
		return h("button", { type: "button", role: "radio", class: "format-option", "aria-checked": String(t === gguf), onclick: () => { gguf = t; drawFormats(); } },
			h("span", { class: "radio-dot" }), h("span", {}, h("b", {}, name), h("span", { class: "muted small" }, text)));
	}));
	drawFormats();
	const name = h("input", { type: "text", class: "code", value: run.model_name, spellcheck: "false" });
	const keep = h("input", { type: "checkbox" });
	const canOllama = Boolean(hw && hw.ollama);
	const ollama = h("input", { type: "checkbox", checked: canOllama, disabled: !canOllama });
	const go = button("Export", { icon: "box", kind: "primary", size: "lg", disabled: Boolean(job) });
	go.addEventListener("click", () => startExport({ gguf, ollama_name: name.value.trim(), keep_merged: keep.checked, ollama: ollama.checked }, go));
	return h("section", { class: "card card-pad stack-lg" },
		h("h2", { class: "section-title" }, "Export options"),
		h("div", { class: "field" }, h("span", { class: "field-label" }, "Model file"), formatList),
		field("Model name", name, { help: "Used for the Ollama tag and file names. Lowercase letters, digits and . _ : / -" }),
		h("div", { class: "stack" },
			h("label", { class: "check" }, ollama, canOllama ? "Register with Ollama when done" : "Register with Ollama (Ollama isn't installed here)"),
			h("label", { class: "check" }, keep, "Also keep the merged Hugging Face model (for transformers or vLLM)"),
		),
		h("div", { class: "row" }, go, job ? h("span", { class: "faint" }, `Waiting for the ${job.kind} job to finish`) : null),
	);
}

function statusCard(run) {
	const exp = run.export;
	if (!exp) return null;
	const exporting = exp.state === "running" || exp.state === "queued";
	const log = h("pre", { class: "code-block log nowrap" }, run.export_log || "(empty)");
	return h("section", { class: "card card-pad stack" },
		h("div", { class: "spread" },
			h("div", { class: "row" }, h("h2", { class: "section-title" }, "Export"), statePill(exp), exporting && exp.stage ? h("span", { class: "muted" }, exp.stage) : null),
			exporting ? button("Stop", { icon: "stop", kind: "secondary", onclick: () => app.stopRun(run.id) }) : null,
		),
		exporting ? h("div", { class: "progress indeterminate" }, h("span")) : null,
		exp.error ? callout("bad", "alert-circle", exp.error) : null,
		(exp.notes || []).map((note) => callout("info", "info", note)),
		h("details", { open: exporting }, h("summary", { class: "muted" }, "Export log"), log),
	);
}

function filesCard(run) {
	if (!run.files.length) return null;
	const shown = run.files.filter((f) => !f.path.startsWith("merged/"));
	const merged = run.files.filter((f) => f.path.startsWith("merged/"));
	const rows = shown.map((f) => h("tr", {}, h("td", {}, h("code", {}, f.path)), h("td", { class: "num" }, bytes(f.bytes))));
	if (merged.length) rows.push(h("tr", {}, h("td", {}, h("code", {}, `merged/ (${merged.length} files)`)), h("td", { class: "num" }, bytes(merged.reduce((a, f) => a + f.bytes, 0)))));
	return h("section", { class: "stack" },
		h("div", { class: "spread" },
			h("h2", { class: "section-title" }, "Files"),
			button("Open folder", { icon: "folder", kind: "secondary", size: "sm", onclick: openFolder }),
		),
		h("div", { class: "table-wrap" }, h("table", {}, h("tbody", {}, rows))),
	);
}

async function openFolder() {
	try {
		const { path } = await api(`/projects/${S.pid}/runs/${S.runId}/open`, { method: "POST" });
		toast(`Opened ${path}`);
	} catch (e) {
		toastError(e);
	}
}

async function renderSnippets(box) {
	if (!S.snippets) {
		try {
			S.snippets = (await api(`/projects/${S.pid}/runs/${S.runId}/snippets`)).snippets;
		} catch (e) {
			mount(box, callout("bad", "alert-circle", e.message));
			return;
		}
	}
	const code = S.snippets[S.snippetTab] || "";
	mount(box,
		h("div", { class: "spread" },
			h("div", { class: "tabs", role: "tablist" }, SNIPPETS.map(([k, label]) => h("button", {
				type: "button", role: "tab", "aria-selected": String(k === S.snippetTab), onclick: () => { S.snippetTab = k; renderSnippets(box); },
			}, label))),
			button("Copy", { icon: "copy", kind: "ghost", size: "sm", onclick: async () => { await navigator.clipboard.writeText(code); toast("Copied"); } }),
		),
		h("pre", { class: "code-block nowrap snippet" }, code),
	);
}

export function renderExport() {
	const pane = $("#tab-export");
	const completed = S.runs.filter((r) => r.status.state === "completed");
	if (!completed.length) {
		mount(pane, emptyState("box", "Nothing to export yet", "Train a model first. Completed runs can be exported here.",
			button("Go to training", { kind: "secondary", onclick: () => app.showTab("train") })));
		return;
	}
	if (!S.run || !completed.some((r) => r.id === S.runId)) {
		selectRun(completed[0].id).then(renderExport, toastError);
		return;
	}
	const run = S.run;
	const select = h("select", { "aria-label": "Run to export" }, completed.map((r) => h("option", { value: r.id, selected: r.id === S.runId },
		`${when(r.created) || r.id}, ${shortModel(r.base_model)}, exact match ${pct(r.metrics && r.metrics.exact_match)}`)));
	select.addEventListener("change", async () => {
		try { await selectRun(select.value); } catch (e) { toastError(e); }
		renderExport();
	});
	const snippetBox = h("div", { class: "stack" });
	mount(pane, h("div", { class: "export-grid" },
		h("div", { class: "stack-lg" },
			field("Run", select),
			optionsCard(run),
			statusCard(run),
		),
		h("div", { class: "stack-lg" },
			filesCard(run),
			h("section", { class: "stack" },
				h("h2", { class: "section-title" }, "Use it from your code"),
				h("p", { class: "muted" }, run.input_kind === "image"
					? "Serve the GGUF and its image encoder (the mmproj file) with Ollama or llama.cpp's llama-server, then send each image with the request. The clients pass the schema so the server constrains output to valid JSON."
					: "Serve the GGUF with Ollama or llama.cpp's llama-server, then call it over HTTP. The clients pass the schema so the server constrains output to valid JSON."),
				snippetBox,
			),
		),
	));
	renderSnippets(snippetBox);
}
