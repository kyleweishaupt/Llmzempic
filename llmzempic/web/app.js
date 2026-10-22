"use strict";

// ---------- helpers ----------

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function h(tag, attrs, ...children) {
	const el = document.createElement(tag);
	for (const [k, v] of Object.entries(attrs || {})) {
		if (v === null || v === undefined || v === false) continue;
		if (k === "class") el.className = v;
		else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
		else if (k === "value") el.value = v;
		else if (k === "checked" || k === "disabled" || k === "selected" || k === "open") el[k] = Boolean(v);
		else el.setAttribute(k, v === true ? "" : v);
	}
	for (const c of children.flat()) {
		if (c === null || c === undefined || c === false) continue;
		el.append(c instanceof Node ? c : document.createTextNode(String(c)));
	}
	return el;
}

const SVG_NS = "http://www.w3.org/2000/svg";
function s(tag, attrs, ...children) {
	const el = document.createElementNS(SVG_NS, tag);
	for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
	for (const c of children) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
	return el;
}

async function api(path, { method = "GET", body } = {}) {
	const opts = { method, headers: {} };
	if (body !== undefined) {
		opts.headers["Content-Type"] = "application/json";
		opts.body = JSON.stringify(body);
	}
	const res = await fetch("/api" + path, opts);
	const data = await res.json().catch(() => ({}));
	if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : res.statusText);
	return data;
}

let toastTimer;
function toast(message, error = false) {
	const el = $("#toast");
	el.textContent = message;
	el.className = "show" + (error ? " error" : "");
	clearTimeout(toastTimer);
	toastTimer = setTimeout(() => (el.className = ""), error ? 6000 : 2500);
}

const pct = (v) => (v === null || v === undefined ? "-" : `${Math.round(v * 100)}%`);
const clip = (text, n) => (text.length > n ? text.slice(0, n) + "…" : text);
const jsonText = (v) => {
	const one = JSON.stringify(v);
	return one && one.length <= 80 ? one : JSON.stringify(v, null, 2);
};
function bytes(n) {
	const units = ["B", "KB", "MB", "GB"];
	let i = 0;
	while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
	return `${n.toFixed(i ? 1 : 0)} ${units[i]}`;
}
function duration(sec) {
	if (sec === null || sec === undefined) return "";
	sec = Math.round(sec);
	if (sec < 60) return `${sec}s`;
	if (sec < 3600) return `${Math.floor(sec / 60)}m ${sec % 60}s`;
	return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
}
const when = (ts) => (ts ? new Date(ts * 1000).toLocaleString() : "");
const randomId = () => Math.random().toString(16).slice(2, 12);

const STATE_ICON = { completed: "✓", failed: "✕", stopped: "■", running: "●", queued: "○" };
function stateBadge(status) {
	const state = (status && status.state) || "unknown";
	return h("span", { class: `badge ${state}` }, `${STATE_ICON[state] || "?"} ${state}`);
}

// Debounced saves keyed by kind; flushed before switching project or starting a job.
const saver = {
	timers: {},
	schedule(key, fn, ms = 600) {
		if (this.timers[key]) clearTimeout(this.timers[key].t);
		const entry = { fn };
		entry.t = setTimeout(() => { delete this.timers[key]; fn(); }, ms);
		this.timers[key] = entry;
	},
	async flush() {
		const entries = Object.values(this.timers);
		this.timers = {};
		for (const e of entries) {
			clearTimeout(e.t);
			await e.fn();
		}
	},
};

// ---------- state ----------

const S = {
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
	savedSchema: "",
	page: 0,
	filter: "all",
	runs: [],
	runId: null,
	run: null,
	events: [],
	tab: "task",
	snippets: null,
	snippetTab: "php",
	logOpen: false,
	mistakesOnly: false,
	tryInput: "",
};
const PAGE_SIZE = 25;

// ---------- header, sidebar ----------

function renderHeader() {
	const hw = S.system && S.system.hardware;
	const el = $("#hw");
	if (!hw) { el.textContent = "Detecting hardware…"; return; }
	const parts = [];
	if (hw.device === "cuda") parts.push(`GPU: ${hw.gpus.map((g) => `${g.name} (${g.vram_gb} GB)`).join(", ")}`);
	else if (hw.device === "mps") parts.push("Apple GPU (MPS)");
	else parts.push("No GPU - CPU only");
	if (hw.ram_gb) parts.push(`${hw.ram_gb} GB RAM`);
	el.textContent = parts.join(" · ");
	if (!hw.can_train) el.append(h("span", { class: "badge critical" }, " · ✕ training libraries missing"));
}

const JOB_LABEL = { train: "Training", export: "Exporting", predict: "Trying a model" };

function renderJobBanner() {
	const el = $("#job-banner");
	el.replaceChildren();
	const job = S.system && S.system.active_job;
	if (!job) return;
	const project = S.projects.find((p) => p.id === job.project_id);
	el.append(h("div", { class: "card" },
		h("b", {}, JOB_LABEL[job.kind] || job.kind), " ",
		h("span", { class: "muted" }, project ? project.name : job.project_id),
		h("div", { class: "muted" }, `running ${duration(Date.now() / 1000 - job.started)}`),
		h("button", { class: "ghost small", type: "button", onclick: () => openProject(job.project_id, job.kind === "export" ? "export" : "train", job.run_id) }, "View"),
	));
}

async function loadProjects() {
	S.projects = (await api("/projects")).projects;
	renderProjects();
}

function renderProjects() {
	const list = $("#project-list");
	list.replaceChildren(...S.projects.map((p) => {
		const last = p.last_run;
		const sub = last ? `${STATE_ICON[last.status.state] || ""} ${last.status.state} · ${p.base_model}` : p.base_model;
		return h("li", { class: p.id === S.pid ? "active" : "", onclick: () => openProject(p.id) },
			p.name, h("span", { class: "sub" }, sub));
	}));
	$("#welcome").hidden = Boolean(S.pid);
	$("#project").hidden = !S.pid;
}

// ---------- project ----------

async function openProject(pid, tab, runId) {
	await saver.flush();
	if (pid !== S.pid) {
		const [view, ex, runs] = await Promise.all([
			api(`/projects/${pid}`), api(`/projects/${pid}/examples`), api(`/projects/${pid}/runs`),
		]);
		S.pid = pid;
		S.view = view;
		S.config = view.config;
		S.savedSchema = JSON.stringify(view.config.schema);
		setExamples(ex);
		S.runs = runs.runs;
		S.page = 0;
		S.runId = null;
		S.run = null;
		S.events = [];
		S.snippets = null;
		S.tryInput = "";
		$("#import-warnings").replaceChildren();
	}
	if (runId) await selectRun(runId);
	else if (!S.runId && S.runs.length) await selectRun(S.runs[0].id);
	$("#project-title").textContent = S.config.name || pid;
	setSaveState("");
	renderProjects();
	showTab(tab || S.tab);
}

function setSaveState(text) {
	$("#save-state").textContent = text;
}

function scheduleConfigSave() {
	const pid = S.pid;
	const config = structuredClone(S.config);
	setSaveState("Unsaved changes…");
	saver.schedule("config", async () => {
		try {
			const view = await api(`/projects/${pid}`, { method: "PUT", body: { config } });
			if (S.pid !== pid) return;
			S.view = view;
			setSaveState("Saved");
			renderSchemaStatus();
			const schema = JSON.stringify(config.schema);
			if (schema !== S.savedSchema) {
				S.savedSchema = schema;
				refreshExampleChecks(pid).catch((e) => toast(e.message, true));
			}
			const p = S.projects.find((x) => x.id === pid);
			if (p && (p.name !== config.name || p.base_model !== config.base_model)) {
				p.name = config.name;
				p.base_model = config.base_model;
				renderProjects();
			}
		} catch (e) {
			setSaveState("Not saved");
			toast(e.message, true);
		}
	});
}

function showTab(tab) {
	S.tab = tab;
	history.replaceState(null, "", `#${S.pid}/${tab}`);
	for (const b of $$("#tabs button")) b.classList.toggle("active", b.dataset.tab === tab);
	for (const pane of $$(".tab")) pane.hidden = pane.id !== `tab-${tab}`;
	({ task: renderTask, examples: renderExamples, model: renderModel, train: renderTrain, export: renderExport })[tab]();
	renderExampleCount();
}

// ---------- 1. task ----------

function renderTask() {
	$("#cfg-name").value = S.config.name || "";
	$("#cfg-instructions").value = S.config.instructions || "";
	$("#cfg-schema").value = JSON.stringify(S.config.schema, null, 2);
	$("#cfg-include-schema").checked = S.config.include_schema_in_prompt !== false;
	renderSchemaStatus();
}

function renderSchemaStatus(localError) {
	const errors = localError ? [localError] : (S.view.schema_errors || []);
	$("#schema-errors").replaceChildren(...errors.map((e) => h("div", {}, e)));
	$("#system-prompt").textContent = S.view.system_prompt;
}

function bindTask() {
	$("#cfg-name").addEventListener("input", (e) => {
		S.config.name = e.target.value;
		$("#project-title").textContent = e.target.value || S.pid;
		scheduleConfigSave();
	});
	$("#cfg-instructions").addEventListener("input", (e) => {
		S.config.instructions = e.target.value;
		scheduleConfigSave();
	});
	$("#cfg-schema").addEventListener("input", (e) => {
		try {
			const schema = JSON.parse(e.target.value);
			if (typeof schema !== "object" || schema === null || Array.isArray(schema)) throw new Error("The schema must be a JSON object.");
			S.config.schema = schema;
			renderSchemaStatus("");
			scheduleConfigSave();
		} catch (err) {
			renderSchemaStatus(`Not valid JSON yet: ${err.message}`);
		}
	});
	$("#cfg-include-schema").addEventListener("change", (e) => {
		S.config.include_schema_in_prompt = e.target.checked;
		scheduleConfigSave();
	});
	$("#template-select").addEventListener("change", (e) => {
		const t = S.catalog.templates.find((x) => x.id === e.target.value);
		e.target.value = "";
		if (!t || !confirm(`Replace the instructions and schema with the "${t.name}" template?`)) return;
		S.config.instructions = t.instructions;
		S.config.schema = structuredClone(t.schema);
		renderTask();
		scheduleConfigSave();
	});
	$("#infer-schema").addEventListener("click", async () => {
		try {
			await saver.flush();
			const { schema } = await api(`/projects/${S.pid}/infer-schema`, { method: "POST" });
			S.config.schema = schema;
			renderTask();
			scheduleConfigSave();
			toast("Schema inferred from example outputs - review the enums and required fields.");
		} catch (e) {
			toast(e.message, true);
		}
	});
}

// ---------- 2. examples ----------

function setExamples(data) {
	S.examples = data.examples;
	S.invalid = new Map(data.invalid.map((i) => [i.id, i.errors]));
	S.stats = data.stats;
	S.badJson = new Map();
}

// Schema checks and label counts depend on the schema; refresh them without touching unsaved example edits.
async function refreshExampleChecks(pid) {
	const data = await api(`/projects/${pid}/examples`);
	if (S.pid !== pid) return;
	S.invalid = new Map(data.invalid.map((i) => [i.id, i.errors]));
	S.stats = data.stats;
	updateExampleErrors();
	renderDatasetOverview();
}

function exampleErrors(id) {
	if (S.badJson.has(id)) return [S.badJson.get(id)];
	return S.invalid.get(id) || [];
}

function problemCount() {
	return S.examples.filter((e) => exampleErrors(e.id).length).length;
}

function renderExampleCount() {
	const bad = problemCount();
	$("#examples-count").textContent = bad ? `${S.examples.length} · ${bad} ✕` : String(S.examples.length);
}

function skeleton(schema) {
	if (!schema || typeof schema !== "object") return null;
	if (schema.enum) return schema.enum[0];
	const type = Array.isArray(schema.type) ? schema.type.find((t) => t !== "null") : schema.type;
	switch (type) {
		case "object": {
			const out = {};
			for (const [k, v] of Object.entries(schema.properties || {})) out[k] = skeleton(v);
			return out;
		}
		case "array": return [];
		case "string": return "";
		case "integer":
		case "number": return 0;
		case "boolean": return false;
		default: return null;
	}
}

function scheduleExamplesSave() {
	const pid = S.pid;
	const examples = S.examples;
	saver.schedule("examples", async () => {
		try {
			const data = await api(`/projects/${pid}/examples`, { method: "PUT", body: { examples } });
			if (S.pid !== pid) return;
			S.invalid = new Map(data.invalid.map((i) => [i.id, i.errors]));
			S.stats = data.stats;
			updateExampleErrors();
			renderDatasetOverview();
		} catch (e) {
			toast(e.message, true);
		}
	});
}

function visibleExamples() {
	if (S.filter === "invalid") return S.examples.filter((e) => exampleErrors(e.id).length);
	if (S.filter === "duplicates") {
		// Copies of the same input are listed next to each other.
		const byId = new Map(S.examples.map((e) => [e.id, e]));
		return ((S.stats && S.stats.duplicates) || []).flatMap((g) => g.ids).map((id) => byId.get(id)).filter(Boolean);
	}
	return S.examples;
}

const labelText = (v) => (typeof v === "string" ? v : JSON.stringify(v));

function duplicateCounts() {
	const groups = (S.stats && S.stats.duplicates) || [];
	const conflicts = groups.filter((g) => g.conflict).length;
	return { conflicts, plain: groups.length - conflicts };
}

function renderDatasetOverview() {
	const stats = S.stats;
	const el = $("#dataset-overview");
	if (!stats || !stats.count) { el.replaceChildren(); return; }
	const rows = stats.fields.map((f) => h("div", { class: "overview-row" },
		h("b", {}, f.field),
		h("span", { class: "muted" }, f.kind === "tags" ? " examples per tag" : ""),
		h("div", { class: "chips" }, f.values.map((v) => h("span", {
			class: "chip" + (v.count === 0 ? " critical" : v.count < 3 ? " warning" : ""),
			title: v.count === 0 ? "No examples: the model can't learn this value" : v.count < 3 ? "Very few examples" : null,
		}, labelText(v.value), h("span", { class: "n" }, String(v.count))))),
	));
	const { conflicts, plain } = duplicateCounts();
	if (conflicts || plain) {
		rows.push(h("div", { class: "overview-row" },
			h("span", { class: `badge ${conflicts ? "critical" : "warning"}` },
				`${conflicts + plain} input(s) appear more than once` + (conflicts ? `, ${conflicts} with different expected outputs` : "")),
			" ",
			h("button", { class: "ghost small", type: "button", onclick: () => {
				S.filter = "duplicates";
				$("#examples-filter").value = "duplicates";
				S.page = 0;
				renderExamples();
			} }, "Show them"),
		));
	}
	el.replaceChildren(...(rows.length ? [h("div", { class: "card overview" }, rows)] : []));
}

function renderExamples() {
	const list = visibleExamples();
	const pages = Math.max(1, Math.ceil(list.length / PAGE_SIZE));
	S.page = Math.min(S.page, pages - 1);
	const start = S.page * PAGE_SIZE;
	$("#examples-list").replaceChildren(...list.slice(start, start + PAGE_SIZE).map((ex) => exampleCard(ex, S.examples.indexOf(ex))));
	if (!S.examples.length) {
		$("#examples-list").append(h("div", { class: "card muted" }, "No examples yet. Add a few by hand or import a file. Aim for at least 20; 100+ covering every label and edge case is better."));
	}
	renderDatasetOverview();
	$("#examples-pager").replaceChildren(
		pages > 1 ? h("button", { class: "ghost small", type: "button", disabled: S.page === 0, onclick: () => { S.page--; renderExamples(); } }, "‹ Prev") : null,
		pages > 1 ? h("span", { class: "muted" }, `Page ${S.page + 1} of ${pages}`) : null,
		pages > 1 ? h("button", { class: "ghost small", type: "button", disabled: S.page >= pages - 1, onclick: () => { S.page++; renderExamples(); } }, "Next ›") : null,
	);
	updateExampleErrors();
}

function exampleCard(ex, index) {
	const output = h("textarea", { class: "code", rows: 3, spellcheck: "false", value: jsonText(ex.output) });
	output.addEventListener("input", () => {
		try {
			ex.output = JSON.parse(output.value);
			S.badJson.delete(ex.id);
		} catch (err) {
			S.badJson.set(ex.id, `Output is not valid JSON: ${err.message}`);
		}
		updateExampleErrors();
		scheduleExamplesSave();
	});
	const input = h("textarea", { rows: 3, value: ex.input });
	input.addEventListener("input", () => {
		ex.input = input.value;
		scheduleExamplesSave();
	});
	return h("div", { class: "card example", "data-id": ex.id },
		h("label", {}, `Input #${index + 1}`, input),
		h("label", {}, "Expected output (JSON)", output),
		h("div", { class: "actions" },
			h("button", { class: "ghost small", type: "button", title: "Replace the input with a text file's contents", onclick: () => loadInputFromFile(ex) }, "From file…"),
			h("button", { class: "ghost small danger", type: "button", onclick: () => {
				S.examples.splice(S.examples.indexOf(ex), 1);
				S.invalid.delete(ex.id);
				S.badJson.delete(ex.id);
				renderExamples();
				scheduleExamplesSave();
			} }, "Delete"),
		),
		h("div", { class: "errors" }),
	);
}

function updateExampleErrors() {
	for (const card of $$("#examples-list .example")) {
		const errors = exampleErrors(card.dataset.id);
		card.classList.toggle("bad", errors.length > 0);
		$(".errors", card).replaceChildren(...errors.map((e) => h("div", {}, e)));
	}
	const bad = problemCount();
	$("#examples-summary").textContent = `${S.examples.length} examples` + (bad ? ` · ${bad} with problems` : " · all match the schema");
	renderExampleCount();
}

let fileTarget = null;
function loadInputFromFile(ex) {
	fileTarget = ex;
	$("#input-file").value = "";
	$("#input-file").click();
}

function bindExamples() {
	$("#add-example").addEventListener("click", () => {
		S.examples.unshift({ id: randomId(), input: "", output: skeleton(S.config.schema) });
		S.page = 0;
		S.filter = "all";
		$("#examples-filter").value = "all";
		renderExamples();
		scheduleExamplesSave();
		$("#examples-list textarea").focus();
	});
	$("#examples-filter").addEventListener("change", (e) => {
		S.filter = e.target.value;
		S.page = 0;
		renderExamples();
	});
	$("#import-btn").addEventListener("click", () => {
		$("#import-file").value = "";
		$("#import-file").click();
	});
	$("#import-file").addEventListener("change", async (e) => {
		const file = e.target.files[0];
		if (!file) return;
		try {
			await saver.flush();
			const text = await file.text();
			const data = await api(`/projects/${S.pid}/import`, {
				method: "POST",
				body: { text, filename: file.name, mode: $("#import-mode").value },
			});
			setExamples(data);
			S.page = 0;
			renderExamples();
			$("#import-warnings").replaceChildren(...data.warnings.slice(0, 20).map((w) => h("div", {}, w)));
			toast(`Imported ${data.imported} example(s)` + (data.warnings.length ? `, ${data.warnings.length} skipped` : ""));
		} catch (err) {
			toast(err.message, true);
		}
	});
	$("#input-file").addEventListener("change", async (e) => {
		const file = e.target.files[0];
		if (!file || !fileTarget) return;
		if (file.size > 512 * 1024 && !confirm(`${file.name} is ${bytes(file.size)}. Long inputs need a lot of memory to train on. Use it anyway?`)) return;
		fileTarget.input = await file.text();
		renderExamples();
		scheduleExamplesSave();
	});
	$("#download-examples").addEventListener("click", () => {
		const text = S.examples.map((e) => JSON.stringify({ input: e.input, output: e.output })).join("\n") + "\n";
		const a = h("a", { href: URL.createObjectURL(new Blob([text], { type: "application/jsonl" })), download: `${S.pid}-examples.jsonl` });
		a.click();
		URL.revokeObjectURL(a.href);
	});
}

// ---------- 3. model ----------

const FIT_LABEL = { good: "✓ fits", tight: "! tight", slow: "! slow", "too-big": "✕ too big" };

const TRAINING_FIELDS = [
	{ key: "epochs", label: "Epochs", type: "number", step: 1, min: 1, help: "Passes over the examples" },
	{ key: "learning_rate", label: "Learning rate", type: "number", step: 0.00001, min: 0 },
	{ key: "lora_r", label: "LoRA rank", type: "number", step: 1, min: 1, help: "Higher learns more, uses more memory" },
	{ key: "lora_alpha", label: "LoRA alpha", type: "number", step: 1, min: 1 },
	{ key: "lora_dropout", label: "LoRA dropout", type: "number", step: 0.01, min: 0 },
	{ key: "batch_size", label: "Batch size", type: "number", step: 1, min: 1, help: "Lower it if you run out of memory" },
	{ key: "grad_accum", label: "Gradient accumulation", type: "number", step: 1, min: 1 },
	{ key: "max_seq_length", label: "Max tokens per example", type: "number", step: 64, min: 0, help: "0 = fit the longest example" },
	{ key: "eval_fraction", label: "Held-out fraction", type: "number", step: 0.05, min: 0, help: "Used for scoring (10+ examples)" },
	{ key: "quantization", label: "Quantization", type: "select", options: [["auto", "Auto"], ["4bit", "4-bit QLoRA"], ["none", "None (16-bit)"]] },
	{ key: "accelerator", label: "Accelerator", type: "select", options: [["auto", "Auto (Unsloth if installed)"], ["unsloth", "Unsloth"], ["off", "Standard PEFT"]] },
	{ key: "seed", label: "Random seed", type: "number", step: 1, min: 0 },
];

function renderModel() {
	const hw = S.system && S.system.hardware;
	const hint = $("#model-hw");
	if (hw) {
		if (hw.device === "cuda") {
			hint.textContent = `Training on ${hw.gpus[0].name} with ${hw.vram_gb} GB VRAM` +
				` · 4-bit QLoRA ${hw.can_4bit ? "available" : "unavailable (pip install bitsandbytes)"}` +
				` · Unsloth ${hw.can_unsloth ? "available (faster training)" : "not installed"}`;
		} else {
			hint.textContent = "No NVIDIA GPU detected, so training runs on the CPU: fine for tiny models and quick tests, very slow for 1B+. Exported models still run well on CPUs.";
		}
	}
	const rows = [];
	let tier = null;
	for (const m of S.catalog.models) {
		if (m.tier !== tier) {
			tier = m.tier;
			rows.push(h("tr", { class: "tier-row" }, h("td", { colspan: 5 }, tier)));
		}
		const selected = m.id === S.config.base_model;
		rows.push(h("tr", { class: "selectable" + (selected ? " selected" : ""), onclick: () => selectModel(m.id) },
			h("td", {}, h("input", { type: "radio", name: "model", checked: selected }), " "),
			h("td", {}, h("b", {}, m.name), m.recommended ? h("span", { class: "pill" }, "recommended") : null, h("span", { class: "sub" }, m.id)),
			h("td", { class: "num" }, `${m.params_b} B`, h("span", { class: "sub" }, m.license + (m.gated ? " · gated" : ""))),
			h("td", {}, h("span", { class: `badge ${m.fit}` }, FIT_LABEL[m.fit] || m.fit), h("span", { class: "sub" }, m.fit_detail)),
			h("td", {}, h("span", { class: "sub" }, m.notes)),
		));
	}
	$("#model-table").replaceChildren(h("table", {},
		h("thead", {}, h("tr", {}, h("th", {}, ""), h("th", {}, "Model"), h("th", {}, "Size"), h("th", {}, "This machine"), h("th", {}, "Good for"))),
		h("tbody", {}, rows),
	));
	$("#cfg-base-model").value = S.config.base_model || "";
	renderTraining();
}

function selectModel(id) {
	S.config.base_model = id;
	scheduleConfigSave();
	renderModel();
}

function currentPreset() {
	const t = S.config.training;
	for (const [name, values] of Object.entries(S.catalog.presets)) {
		if (Object.entries(values).every(([k, v]) => Number(t[k]) === v)) return name;
	}
	return "custom";
}

function renderTraining() {
	const active = currentPreset();
	const buttons = Object.keys(S.catalog.presets).map((name) =>
		h("button", { type: "button", class: name === active ? "active" : "", onclick: () => {
			Object.assign(S.config.training, S.catalog.presets[name], { preset: name });
			scheduleConfigSave();
			renderTraining();
		} }, name[0].toUpperCase() + name.slice(1)));
	if (active === "custom") buttons.push(h("button", { type: "button", class: "active", disabled: true }, "Custom"));
	$("#presets").replaceChildren(...buttons);
	$("#training-fields").replaceChildren(...TRAINING_FIELDS.map((f) => {
		const value = S.config.training[f.key];
		const control = f.type === "select"
			? h("select", {}, f.options.map(([v, label]) => h("option", { value: v, selected: v === value }, label)))
			: h("input", { type: "number", step: f.step, min: f.min, value: value });
		control.addEventListener("change", () => {
			S.config.training[f.key] = f.type === "select" ? control.value : Number(control.value);
			scheduleConfigSave();
			renderTraining();
		});
		return h("label", {}, f.label, control, f.help ? h("span", { class: "sub muted" }, f.help) : null);
	}));
}

function bindModel() {
	$("#cfg-base-model").addEventListener("change", (e) => selectModel(e.target.value.trim()));
}

// ---------- 4. train ----------

async function loadRuns() {
	const data = await api(`/projects/${S.pid}/runs`);
	S.runs = data.runs;
}

async function selectRun(runId) {
	const data = await api(`/projects/${S.pid}/runs/${runId}`);
	S.runId = runId;
	S.run = data;
	S.events = data.events;
	S.snippets = null;
}

async function pollRun() {
	const data = await api(`/projects/${S.pid}/runs/${S.runId}?since=${S.events.length}&items=false`);
	S.events.push(...data.events);
	data.evaluation = S.run && S.run.evaluation;
	S.run = data;
}

function readiness() {
	const hw = S.system && S.system.hardware;
	const bad = problemCount();
	const n = S.examples.length;
	const checks = [
		[S.view.schema_errors.length === 0, "Output schema is valid", "Fix the output schema on the Task tab"],
		[n >= 2 && bad === 0, `${n} examples, all matching the schema`, n < 2 ? "Add at least 2 examples" : `${bad} example(s) need fixing on the Examples tab`],
		[Boolean(S.config.base_model), `Base model: ${S.config.base_model}`, "Choose a base model"],
		[Boolean(hw && hw.can_train), "Training libraries installed", `Install the training libraries (${hw ? hw.missing_train_packages.join(", ") || "PyTorch" : "…"}) with the setup script`],
	];
	const warnings = [];
	if (n >= 2 && n < 20) warnings.push("Fewer than 20 examples: results will be unreliable. 100+ is better.");
	if (n >= 2 && n < 10) warnings.push("With fewer than 10 examples, scoring uses training examples (no held-out set).");
	if (n >= 2 && S.stats) {
		for (const f of S.stats.fields) {
			const unseen = f.values.filter((v) => v.count === 0).map((v) => labelText(v.value));
			if (unseen.length) warnings.push(`${f.field}: no examples of ${unseen.join(", ")}. The model can't learn a value it never sees.`);
		}
		const { conflicts, plain } = duplicateCounts();
		if (conflicts) warnings.push(`${conflicts} input(s) appear more than once with different expected outputs; the model can't learn both answers.`);
		if (plain) warnings.push(`${plain} input(s) appear more than once. Copies stay together when examples are held out, but each copy counts extra in training.`);
	}
	if (hw && hw.device !== "cuda") warnings.push("No GPU: training will be slow for models above ~0.5B parameters.");
	return { ok: checks.every((c) => c[0]), checks, warnings };
}

function renderTrain() {
	const r = readiness();
	const job = S.system && S.system.active_job;
	$("#readiness").replaceChildren(
		...r.checks.map(([ok, good, badText]) => h("div", { class: `badge ${ok ? "good" : "critical"}` }, ok ? `✓ ${good}` : `✕ ${badText}`)),
		...r.warnings.map((w) => h("div", { class: "badge warning" }, `! ${w}`)),
	);
	$("#start-training").disabled = !r.ok || Boolean(job);
	$("#train-msg").textContent = job ? `A ${job.kind} job is running.` : "";
	renderRunsTable();
	renderRunDetail();
}

function renderRunsTable() {
	if (!S.runs.length) {
		$("#runs-table").replaceChildren(h("p", { class: "muted" }, "No runs yet."));
		return;
	}
	$("#runs-table").replaceChildren(h("table", {},
		h("thead", {}, h("tr", {}, ["Run", "Base model", "Status", "Schema valid", "Exact match", "Export", ""].map((t) => h("th", {}, t)))),
		h("tbody", {}, S.runs.map((run) => h("tr", {
			class: "selectable" + (run.id === S.runId ? " selected" : ""),
			onclick: async () => { await selectRun(run.id); renderTrain(); },
		},
			h("td", {}, run.id, h("span", { class: "sub" }, `${run.train_count} train / ${run.eval_count} held out` +
				(run.eval_count && run.split && run.split.stratified_by ? ` · balanced by ${run.split.stratified_by}` : ""))),
			h("td", {}, run.base_model),
			h("td", {}, stateBadge(run.status), h("span", { class: "sub" }, run.status.state === "running" ? run.status.stage : "")),
			h("td", { class: "num" }, pct(run.metrics && run.metrics.schema_valid)),
			h("td", { class: "num" }, pct(run.metrics && run.metrics.exact_match)),
			h("td", {}, run.export ? stateBadge(run.export) : ""),
			h("td", {}, h("button", { class: "ghost small danger", type: "button", onclick: (e) => { e.stopPropagation(); deleteRun(run.id); } }, "Delete")),
		))),
	));
}

async function deleteRun(runId) {
	if (!confirm(`Delete run ${runId} and its files?`)) return;
	try {
		await api(`/projects/${S.pid}/runs/${runId}`, { method: "DELETE" });
		if (S.runId === runId) { S.runId = null; S.run = null; S.events = []; }
		await loadRuns();
		renderTrain();
	} catch (e) {
		toast(e.message, true);
	}
}

function renderRunDetail() {
	const el = $("#run-detail");
	const run = S.run;
	if (!run) { el.replaceChildren(); return; }
	const st = run.status || {};
	const running = st.state === "running" || st.state === "queued";
	const parts = [h("h3", {}, `Run ${run.id} `, stateBadge(st), " ", h("span", { class: "muted" }, st.stage || ""))];
	if (st.plan) {
		parts.push(h("div", { class: "muted" }, `${st.plan.device.toUpperCase()} · ${st.plan.precision}` +
			(st.plan.four_bit ? " · 4-bit QLoRA" : "") + (st.plan.unsloth ? " · Unsloth" : "") +
			(st.train_seconds ? ` · trained in ${duration(st.train_seconds)}` : "")));
	}
	if (running && st.progress && st.progress.total) {
		const p = st.progress;
		parts.push(h("div", { class: "progress" }, h("div", { style: `width:${Math.round((100 * p.step) / p.total)}%` })));
		parts.push(h("div", { class: "muted" }, `Step ${p.step} of ${p.total}` + (p.eta ? ` · about ${duration(p.eta)} left` : "")));
	}
	if (running) parts.push(h("button", { class: "ghost danger", type: "button", onclick: () => stopRun(run.id) }, "Stop"));
	if (st.error) parts.push(h("div", { class: "errors" }, h("div", {}, st.error)));
	const chart = lossChart(S.events);
	if (chart) parts.push(chart);
	const ev = run.evaluation;
	if (ev) parts.push(evaluationView(ev));
	else if (run.metrics) parts.push(metricTiles(run.metrics));
	if (st.state === "completed") {
		parts.push(tryPanel(run));
		parts.push(h("button", { type: "button", onclick: () => showTab("export") }, "Export this model →"));
	}
	const details = h("details", { open: S.logOpen || running }, h("summary", {}, "Training log"), h("pre", { class: "code-block log" }, run.log || "(empty)"));
	details.addEventListener("toggle", () => (S.logOpen = details.open));
	parts.push(details);
	el.replaceChildren(...parts);
	const log = $(".log", el);
	log.scrollTop = log.scrollHeight;
}

// ---------- try a trained run ----------

function tryPanel(run) {
	const pr = run.predict || {};
	const pst = pr.status;
	const job = S.system && S.system.active_job;
	const mine = Boolean(job && job.kind === "predict" && job.run_id === run.id);
	const input = h("textarea", { rows: 4, value: S.tryInput, placeholder: "Paste an input the model hasn't seen…" });
	input.addEventListener("input", () => (S.tryInput = input.value));
	const parts = [
		h("h3", {}, "Try it"),
		h("p", { class: "hint" }, "Run this model on your own input before exporting it. Each try loads the model first, so an answer can take a minute."),
		input,
		h("div", { class: "row" },
			h("button", { type: "button", disabled: Boolean(job), onclick: () => tryModel(run.id) }, mine ? "Running…" : "Run"),
			mine ? h("button", { class: "ghost danger", type: "button", onclick: () => stopRun(run.id) }, "Stop") : null,
			h("span", { class: "muted" }, mine ? (pst && pst.stage) || "Starting" : job ? `Busy: ${JOB_LABEL[job.kind] || job.kind}…` : ""),
		),
	];
	if (!mine && pst && pst.state === "failed") parts.push(h("div", { class: "errors" }, h("div", {}, pst.error || "Failed")));
	if (!mine && pr.result) parts.push(...pr.result.items.map(tryResult));
	return h("div", { class: "card try" }, parts);
}

function tryResult(it) {
	const ok = it.json_valid && !it.errors.length;
	return h("div", { class: "try-result" },
		h("div", { class: "muted" }, "Input: ", clip(it.input, 160)),
		h("pre", { class: "code-block" }, it.json_valid ? JSON.stringify(it.value, null, 2) : it.generated || "(no answer)"),
		h("div", { class: "row-between" },
			h("span", { class: `badge ${ok ? "good" : it.json_valid ? "warning" : "critical"}` }, ok ? "✓ Matches the schema" : `✕ ${it.errors[0]}`),
			h("button", { class: "ghost small", type: "button", title: "Add this input as a training example, with the answer as its expected output", onclick: () => addTryExample(it, ok) }, "Add as example"),
		),
	);
}

function addTryExample(it, ok) {
	S.examples.unshift({ id: randomId(), input: it.input, output: ok ? structuredClone(it.value) : skeleton(S.config.schema) });
	S.page = 0;
	scheduleExamplesSave();
	renderExampleCount();
	renderTrain();
	toast("Added as the first example. Correct its expected output on the Examples tab, then train again.");
}

async function tryModel(runId) {
	if (!S.tryInput.trim()) {
		toast("Enter an input to try.", true);
		return;
	}
	try {
		await api(`/projects/${S.pid}/runs/${runId}/predict`, { method: "POST", body: { inputs: [S.tryInput] } });
		S.system = await api("/system");
		await selectRun(runId);
		renderJobBanner();
		renderTrain();
		schedulePoll(1000);
	} catch (e) {
		toast(e.message, true);
	}
}

function metricTiles(m) {
	return h("div", {},
		h("div", { class: "tiles" },
			[["json_valid", "Valid JSON"], ["schema_valid", "Matches schema"], ["exact_match", "Exact match"]].map(([k, label]) =>
				h("div", { class: "tile" }, h("div", { class: "value" }, pct(m[k])), h("div", { class: "label" }, label))),
		),
		h("div", { class: "muted" }, `Scored on ${m.count} ${m.eval_on || "examples"}. Deployed with schema-constrained decoding, every response is valid JSON; exact match shows how often the content is right.`),
	);
}

function rateCell(hits, total, text) {
	return h("td", { class: "num" + (total && hits / total < 0.5 ? " low" : "") }, total ? text : "-");
}

function labelBreakdown(fields) {
	return (fields || []).map((f) => h("div", { class: "breakdown" },
		h("table", {},
			h("thead", {}, h("tr", {},
				h("th", {}, f.kind === "tags" ? `${f.field} (tags)` : f.field),
				h("th", { class: "num", title: "Scored examples whose expected answer has this value" }, "Examples"),
				h("th", { class: "num", title: "How many of those examples the model answered with this value" }, "Recognized"),
				h("th", { class: "num", title: "When the model answered with this value, how often that was right" }, "Precision"),
			)),
			h("tbody", {}, f.values.map((v) => h("tr", {},
				h("td", {}, labelText(v.value)),
				h("td", { class: "num" }, String(v.expected)),
				rateCell(v.correct, v.expected, `${v.correct} · ${pct(v.correct / v.expected)}`),
				rateCell(v.correct, v.predicted, `${pct(v.correct / v.predicted)} of ${v.predicted}`),
			))),
		),
		f.confusions.length ? h("p", { class: "muted" }, "Most common mix-ups (expected → answered): " + f.confusions.map((c) =>
			`${labelText(c.expected)} → ${c.unknown ? "no valid value" : labelText(c.predicted)} ×${c.count}`).join(" · ")) : null,
	));
}

function predictionRow(it) {
	return h("tr", {},
		h("td", {}, clip(it.input, 200)),
		h("td", { class: "mono" }, jsonText(it.expected)),
		h("td", { class: "mono" }, clip(it.generated, 400)),
		h("td", {}, it.exact_match
			? h("span", { class: "badge good" }, "✓ exact")
			: h("span", { class: `badge ${it.schema_valid ? "warning" : "critical"}` }, it.schema_valid ? "! differs" : "✕ " + (it.errors[0] || "invalid"))),
	);
}

function evaluationView(ev) {
	const fields = Object.entries(ev.field_accuracy || {});
	const wrong = ev.items.filter((it) => !it.exact_match).length;
	const toggle = wrong > 0 && wrong < ev.items.length;
	const body = h("tbody");
	const fill = () => body.replaceChildren(...ev.items.filter((it) => !(toggle && S.mistakesOnly) || !it.exact_match).slice(0, 100).map(predictionRow));
	fill();
	return h("div", {},
		metricTiles(ev),
		fields.length > 1 ? h("p", { class: "muted" }, "Per-field accuracy: " + fields.map(([k, v]) => `${k} ${pct(v)}`).join(" · ")) : null,
		labelBreakdown(ev.labels),
		h("details", {}, h("summary", {}, `Predictions (${ev.items.length}` + (wrong ? `, ${wrong} not exact)` : ")")),
			toggle ? h("label", { class: "check" },
				h("input", { type: "checkbox", checked: S.mistakesOnly, onchange: (e) => { S.mistakesOnly = e.target.checked; fill(); } }),
				" Only show mistakes") : null,
			h("table", {},
				h("thead", {}, h("tr", {}, ["Input", "Expected", "Generated", ""].map((t) => h("th", {}, t)))),
				body,
			),
		),
	);
}

async function stopRun(runId) {
	try {
		await api(`/projects/${S.pid}/runs/${runId}/stop`, { method: "POST" });
		await refreshAfterJob();
	} catch (e) {
		toast(e.message, true);
	}
}

function niceTicks(max, count) {
	const raw = max / count;
	const mag = Math.pow(10, Math.floor(Math.log10(raw)));
	const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
	const ticks = [];
	for (let v = 0; v < max + step - 1e-9; v += step) ticks.push(Number(v.toPrecision(6)));
	return ticks;
}

function lossChart(events) {
	const train = events.filter((e) => e.loss !== undefined).map((e) => [e.step, e.loss]);
	const evals = events.filter((e) => e.eval_loss !== undefined).map((e) => [e.step, e.eval_loss]);
	if (!train.length && !evals.length) return null;
	const W = 800, H = 220, m = { l: 48, r: 14, t: 10, b: 26 };
	const all = train.concat(evals);
	const maxX = Math.max(1, ...all.map((p) => p[0]));
	const ticksY = niceTicks(Math.max(...all.map((p) => p[1])) * 1.05 || 1, 4);
	const maxY = ticksY[ticksY.length - 1] || 1;
	const x = (v) => m.l + (v / maxX) * (W - m.l - m.r);
	const y = (v) => H - m.b - (v / maxY) * (H - m.t - m.b);
	const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Training loss by step" });
	for (const t of ticksY) {
		svg.append(s("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: y(t), y2: y(t) }));
		svg.append(s("text", { class: "axis", x: m.l - 6, y: y(t) + 4, "text-anchor": "end" }, t));
	}
	const stepX = Math.max(1, Math.ceil(maxX / 8));
	for (let v = 0; v <= maxX; v += stepX) {
		svg.append(s("text", { class: "axis", x: x(v), y: H - 8, "text-anchor": "middle" }, v));
	}
	if (train.length) {
		const d = train.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
		svg.append(s("path", { d, fill: "none", stroke: "var(--series-1)", "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
	}
	for (const p of evals) {
		svg.append(s("circle", { cx: x(p[0]), cy: y(p[1]), r: 4.5, fill: "var(--series-2)", stroke: "var(--surface)", "stroke-width": 2 }));
	}
	const cross = s("line", { x1: 0, x2: 0, y1: m.t, y2: H - m.b, stroke: "var(--text-2)", "stroke-width": 1, visibility: "hidden" });
	const dot = s("circle", { r: 4.5, fill: "var(--series-1)", stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden" });
	const hit = s("rect", { x: m.l, y: 0, width: W - m.l - m.r, height: H, fill: "transparent" });
	svg.append(cross, dot, hit);

	const tip = h("div", { class: "tooltip", hidden: true });
	const legend = h("div", { class: "legend" },
		h("span", { style: "--c: var(--series-1)" }, "Training loss"),
		evals.length ? h("span", { class: "dot", style: "--c: var(--series-2)" }, "Held-out loss (per epoch)") : null,
	);
	const wrap = h("div", { class: "chart" }, legend, svg, tip);
	const points = train.length ? train : evals;
	hit.addEventListener("mousemove", (e) => {
		const rect = svg.getBoundingClientRect();
		const vx = ((e.clientX - rect.left) / rect.width) * W;
		let best = points[0];
		for (const p of points) if (Math.abs(x(p[0]) - vx) < Math.abs(x(best[0]) - vx)) best = p;
		cross.setAttribute("x1", x(best[0]));
		cross.setAttribute("x2", x(best[0]));
		dot.setAttribute("cx", x(best[0]));
		dot.setAttribute("cy", y(best[1]));
		cross.setAttribute("visibility", "visible");
		dot.setAttribute("visibility", "visible");
		const evalAt = evals.find((p) => p[0] === best[0]);
		tip.replaceChildren(h("b", {}, `Step ${best[0]}`), h("br"), `${train.length ? "Training" : "Held-out"} loss ${best[1].toFixed(4)}`,
			evalAt && train.length ? h("span", {}, h("br"), `Held-out loss ${evalAt[1].toFixed(4)}`) : null);
		tip.hidden = false;
		const px = (x(best[0]) / W) * rect.width;
		tip.style.left = `${Math.min(px + 16, rect.width - 170)}px`;
		tip.style.top = "36px";
	});
	hit.addEventListener("mouseleave", () => {
		tip.hidden = true;
		cross.setAttribute("visibility", "hidden");
		dot.setAttribute("visibility", "hidden");
	});
	return wrap;
}

function bindTrain() {
	$("#start-training").addEventListener("click", async () => {
		const btn = $("#start-training");
		btn.disabled = true;
		try {
			await saver.flush();
			const run = await api(`/projects/${S.pid}/runs`, { method: "POST" });
			await loadRuns();
			await selectRun(run.id);
			S.system = await api("/system");
			renderJobBanner();
			renderTrain();
			toast("Training started");
			schedulePoll(500);
		} catch (e) {
			toast(e.message, true);
			btn.disabled = false;
		}
	});
}

// ---------- 5. export ----------

const GGUF_LABELS = {
	q8_0: "Q8_0 - near-lossless, ~1 byte per parameter (recommended)",
	q5_k_m: "Q5_K_M - smaller, very close to Q8 quality",
	q4_k_m: "Q4_K_M - smallest and fastest on CPU",
	f16: "F16 - full precision, largest file",
	bf16: "BF16 - full precision, largest file",
	none: "No GGUF (adapter and client files only)",
};

function renderExport() {
	const el = $("#export-body");
	const completed = S.runs.filter((r) => r.status.state === "completed");
	if (!completed.length) {
		el.replaceChildren(h("p", { class: "muted" }, "Train a model first; completed runs can be exported here."));
		return;
	}
	if (!S.run || !completed.some((r) => r.id === S.runId)) {
		selectRun(completed[0].id).then(renderExport);
		return;
	}
	const run = S.run;
	const exp = run.export || null;
	const job = S.system && S.system.active_job;
	const hw = S.system && S.system.hardware;
	const exporting = exp && (exp.state === "running" || exp.state === "queued");

	const runSelect = h("select", {}, completed.map((r) =>
		h("option", { value: r.id, selected: r.id === S.runId }, `${r.id} · ${r.base_model} · exact ${pct(r.metrics && r.metrics.exact_match)}`)));
	runSelect.addEventListener("change", async () => { await selectRun(runSelect.value); renderExport(); });
	const gguf = h("select", {}, S.catalog.gguf_types.map((t) => h("option", { value: t, selected: t === "q8_0" }, GGUF_LABELS[t] || t)));
	const name = h("input", { type: "text", value: run.model_name, spellcheck: "false" });
	const keep = h("input", { type: "checkbox" });
	const ollama = h("input", { type: "checkbox", checked: Boolean(hw && hw.ollama), disabled: !(hw && hw.ollama) });
	const go = h("button", { type: "button", disabled: Boolean(job) }, "Export");
	go.addEventListener("click", async () => {
		go.disabled = true;
		try {
			await api(`/projects/${S.pid}/runs/${S.runId}/export`, {
				method: "POST",
				body: { gguf: gguf.value, ollama_name: name.value.trim(), keep_merged: keep.checked, ollama: ollama.checked },
			});
			S.system = await api("/system");
			await selectRun(S.runId);
			await loadRuns();
			renderJobBanner();
			renderExport();
			schedulePoll(500);
		} catch (e) {
			toast(e.message, true);
			go.disabled = false;
		}
	});

	const parts = [
		h("label", {}, "Run", runSelect),
		h("div", { class: "card" },
			h("label", {}, "Model file format", gguf),
			h("label", {}, "Model name", name),
			h("label", { class: "check" }, keep, "Also keep the merged Hugging Face model (safetensors) for transformers / vLLM"),
			h("label", { class: "check" }, ollama, hw && hw.ollama ? "Register with Ollama when done" : "Register with Ollama (Ollama not found on this machine)"),
			h("div", { class: "row" }, go, job ? h("span", { class: "muted" }, `A ${job.kind} job is running.`) : null),
		),
	];
	if (exp) {
		parts.push(h("h3", {}, "Export ", stateBadge(exp), " ", h("span", { class: "muted" }, exp.stage || "")));
		if (exporting) parts.push(h("button", { class: "ghost danger", type: "button", onclick: () => stopRun(run.id) }, "Stop"));
		if (exp.error) parts.push(h("div", { class: "errors" }, h("div", {}, exp.error)));
		for (const note of exp.notes || []) parts.push(h("div", { class: "badge warning" }, `! ${note}`));
		parts.push(h("details", { open: exporting }, h("summary", {}, "Export log"), h("pre", { class: "code-block log" }, run.export_log || "(empty)")));
	}
	if (run.files.length) {
		parts.push(h("div", { class: "row-between" },
			h("h3", {}, "Files"),
			h("button", { class: "ghost", type: "button", onclick: openFolder }, "Open folder"),
		));
		const shown = run.files.filter((f) => !f.path.startsWith("merged/"));
		const merged = run.files.filter((f) => f.path.startsWith("merged/"));
		const rows = shown.map((f) => h("tr", {}, h("td", { class: "mono" }, f.path), h("td", { class: "num" }, bytes(f.bytes))));
		if (merged.length) rows.push(h("tr", {}, h("td", { class: "mono" }, `merged/ (${merged.length} files)`), h("td", { class: "num" }, bytes(merged.reduce((a, f) => a + f.bytes, 0)))));
		parts.push(h("table", {}, h("tbody", {}, rows)));
	}
	parts.push(h("h3", {}, "Use it from your code"));
	parts.push(h("p", { class: "hint" }, "Serve the GGUF with Ollama (", h("code", {}, "ollama create <name> -f Modelfile"),
		") or llama.cpp's llama-server, then call it over HTTP. The clients pass the schema so the server constrains output to valid JSON."));
	const snippetBox = h("div", { id: "snippets" });
	parts.push(snippetBox);
	el.replaceChildren(...parts);
	renderSnippets(snippetBox);
}

const SNIPPET_TABS = [["php", "PHP"], ["csharp", "C#"], ["python", "Python"], ["llama_server", "llama-server"]];

async function renderSnippets(box) {
	if (!S.snippets) {
		try {
			S.snippets = (await api(`/projects/${S.pid}/runs/${S.runId}/snippets`)).snippets;
		} catch (e) {
			box.replaceChildren(h("div", { class: "errors" }, h("div", {}, e.message)));
			return;
		}
	}
	const code = S.snippets[S.snippetTab];
	box.replaceChildren(
		h("div", { class: "row-between" },
			h("div", { class: "segmented" }, SNIPPET_TABS.map(([k, label]) =>
				h("button", { type: "button", class: k === S.snippetTab ? "active" : "", onclick: () => { S.snippetTab = k; renderSnippets(box); } }, label))),
			h("button", { class: "ghost", type: "button", onclick: async () => {
				await navigator.clipboard.writeText(code);
				toast("Copied");
			} }, "Copy"),
		),
		h("pre", { class: "code-block" }, code),
	);
}

async function openFolder() {
	try {
		const { path } = await api(`/projects/${S.pid}/runs/${S.runId}/open`, { method: "POST" });
		toast(`Opened ${path}`);
	} catch (e) {
		toast(e.message, true);
	}
}

// ---------- polling ----------

let pollTimer = null;
function schedulePoll(ms) {
	clearTimeout(pollTimer);
	pollTimer = setTimeout(poll, ms);
}

async function refreshAfterJob() {
	S.system = await api("/system");
	renderHeader();
	renderJobBanner();
	await loadProjects();
	if (!S.pid) return;
	await loadRuns();
	if (S.runId) await selectRun(S.runId);
	if (S.tab === "train" || S.tab === "export") showTab(S.tab);
}

async function poll() {
	let next = 5000;
	try {
		const before = S.system && S.system.active_job;
		S.system = await api("/system");
		const job = S.system.active_job;
		renderHeader();
		renderJobBanner();
		if (before && !job) {
			await refreshAfterJob();
		} else if (job && S.pid === job.project_id) {
			next = 2000;
			if (job.run_id === S.runId) {
				await pollRun();
				const st = job.kind === "train" ? S.run.status : S.run.export;
				const idx = S.runs.findIndex((r) => r.id === S.runId);
				if (idx >= 0) S.runs[idx] = { ...S.runs[idx], status: S.run.status, export: S.run.export };
				if (S.tab === "train") { renderRunsTable(); renderRunDetail(); }
				if (S.tab === "export" && st) renderExport();
			}
		} else if (job) {
			next = 3000;
		}
	} catch (e) {
		next = 5000;
	}
	schedulePoll(next);
}

// ---------- dialogs & boot ----------

function bindDialogs() {
	const dlg = $("#new-project-dialog");
	const openNew = () => {
		$("#new-project-template").replaceChildren(...S.catalog.templates.map((t) => h("option", { value: t.id }, t.name)));
		$("#new-project-name").value = "";
		dlg.showModal();
	};
	$("#new-project-btn").addEventListener("click", openNew);
	$("#welcome-new").addEventListener("click", openNew);
	$("#new-project-cancel").addEventListener("click", () => dlg.close());
	$("#new-project-form").addEventListener("submit", async (e) => {
		e.preventDefault();
		try {
			const view = await api("/projects", { method: "POST", body: { name: $("#new-project-name").value, template: $("#new-project-template").value } });
			dlg.close();
			await loadProjects();
			await openProject(view.id, "task");
		} catch (err) {
			toast(err.message, true);
		}
	});

	const settings = $("#settings-dialog");
	$("#settings-btn").addEventListener("click", async () => {
		const data = await api("/settings");
		$("#set-hf-token").value = "";
		$("#set-hf-token").placeholder = data.hf_token_set ? "saved - type to replace" : "hf_…";
		$("#set-llama").value = data.llama_cpp_dir || "";
		$("#set-workspace").textContent = `Workspace: ${S.system ? S.system.workspace : ""}`;
		settings.showModal();
	});
	$("#settings-cancel").addEventListener("click", () => settings.close());
	$("#settings-form").addEventListener("submit", async (e) => {
		e.preventDefault();
		const body = { llama_cpp_dir: $("#set-llama").value.trim() };
		if ($("#set-hf-token").value.trim()) body.hf_token = $("#set-hf-token").value.trim();
		try {
			await api("/settings", { method: "PUT", body });
			settings.close();
			toast("Settings saved");
		} catch (err) {
			toast(err.message, true);
		}
	});

	$("#delete-project").addEventListener("click", async () => {
		if (!confirm(`Delete project "${S.config.name}" with all its examples, runs and exports?`)) return;
		try {
			await api(`/projects/${S.pid}`, { method: "DELETE" });
			S.pid = null;
			history.replaceState(null, "", "#");
			await loadProjects();
		} catch (e) {
			toast(e.message, true);
		}
	});

	for (const b of $$("#tabs button")) b.addEventListener("click", () => showTab(b.dataset.tab));
	window.addEventListener("beforeunload", () => { saver.flush(); });
}

async function boot() {
	renderHeader();
	bindTask();
	bindExamples();
	bindModel();
	bindTrain();
	bindDialogs();
	try {
		[S.system, S.catalog] = await Promise.all([api("/system"), api("/catalog")]);
		$("#template-select").append(...S.catalog.templates.map((t) => h("option", { value: t.id }, t.name)));
		renderHeader();
		await loadProjects();
		renderJobBanner();
		const [pid, tab] = location.hash.slice(1).split("/");
		if (pid && S.projects.some((p) => p.id === pid)) await openProject(pid, tab || "task");
	} catch (e) {
		toast(`Could not reach the Llmzempic server: ${e.message}`, true);
	}
	schedulePoll(2000);
}

boot();
