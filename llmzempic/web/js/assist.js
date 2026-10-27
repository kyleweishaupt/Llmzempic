import { api } from "./api.js";
import { button, callout, field, h, mount, spinnerLine, toggleSwitch } from "./dom.js";
import { progressDialog, saver, toast, toastError } from "./feedback.js";
import { plural } from "./format.js";
import { checksKey, loadPage } from "./project.js";
import { S, app } from "./state.js";

// Labeling helpers on the Examples tab: import a Hugging Face dataset, or have a stronger LLM write examples.

function dialog(build) {
	const dlg = h("dialog", {});
	const close = () => dlg.close();
	dlg.addEventListener("close", () => dlg.remove());
	mount(dlg, build(close));
	document.body.append(dlg);
	dlg.showModal();
	return dlg;
}

const select = (options, value) => h("select", {}, options.map(([v, label]) => h("option", { value: v, selected: v === value }, label)));

async function afterImport(view) {
	if (view) Object.assign(S, { view, config: view.config, savedChecks: checksKey(view.config) });
	await loadPage();
	app.renderProjectHead();
	app.renderStepper();
	app.views.examples();
}

export function hubImportDialog() {
	const id = h("input", { type: "text", class: "code", spellcheck: "false", placeholder: "mteb/banking77", value: S.hubDataset || "" });
	const body = h("div", { class: "stack" });
	let info = null;
	const lookUp = async () => {
		S.hubDataset = id.value.trim();
		mount(body, spinnerLine("Asking Hugging Face…"));
		try {
			info = await api(`/hub/dataset?${new URLSearchParams({ id: S.hubDataset })}`);
			renderConfig(info.configs[0]);
		} catch (e) {
			mount(body, callout("bad", "alert-circle", e.message, " Datasets need a parquet version on the Hub (most do; ones with a loading script don't)."));
		}
	};
	const choice = { config: null, split: null, text: null, labels: new Set(), limit: 5000, setSchema: true, replace: false };
	const renderConfig = (config) => {
		Object.assign(choice, { config: config.name, split: (config.splits.find((s) => s.name === "train") || config.splits[0] || {}).name, text: config.text, labels: new Set(config.labels.slice(0, 1)) });
		const configSel = select(info.configs.map((c) => [c.name, c.name]), config.name);
		configSel.addEventListener("change", () => renderConfig(info.configs.find((c) => c.name === configSel.value)));
		const splitSel = select(config.splits.map((s) => [s.name, `${s.name} (${(s.rows || 0).toLocaleString()} rows)`]), choice.split);
		splitSel.addEventListener("change", () => (choice.split = splitSel.value));
		const textSel = select(config.columns.filter((c) => c.kind === "text").map((c) => [c.name, c.name]), choice.text);
		textSel.addEventListener("change", () => (choice.text = textSel.value));
		const limit = h("input", { type: "number", min: 1, max: 50000, step: 500, value: choice.limit });
		limit.addEventListener("change", () => (choice.limit = Number(limit.value) || 5000));
		mount(body,
			info.configs.length > 1 ? field("Config", configSel) : null,
			field("Split", splitSel),
			field("Text column", textSel, { help: "Becomes each example's input." }),
			h("div", { class: "field" }, h("span", { class: "field-label" }, "Label columns"),
				h("div", { class: "stack" }, config.columns.filter((c) => c.name !== config.text && c.kind !== "other").map((c) => toggleSwitch(
					`${c.name}${c.names ? ` (${c.names.length} classes)` : ` (${c.kind})`}`, choice.labels.has(c.name),
					(on) => (on ? choice.labels.add(c.name) : choice.labels.delete(c.name)))))),
			field("Rows to import", limit),
			toggleSwitch("Make the output one field per label column (replaces the output design)", choice.setSchema, (v) => (choice.setSchema = v)),
			toggleSwitch("Replace the current examples", choice.replace, (v) => (choice.replace = v)),
		);
	};
	const run = async (close) => {
		if (!choice.config || !choice.labels.size) { toast("Look up a dataset and choose a label column.", true); return; }
		await saver.flush();
		close();
		const busy = progressDialog("Importing from Hugging Face", `${S.hubDataset}, ${choice.split}: up to ${choice.limit.toLocaleString()} rows…`);
		try {
			const res = await api(`/projects/${S.pid}/hub-import`, { method: "POST", body: {
				dataset: S.hubDataset, config: choice.config, split: choice.split, text: choice.text, labels: [...choice.labels],
				limit: choice.limit, set_schema: choice.setSchema, mode: choice.replace ? "replace" : "append",
			} });
			busy.close();
			await afterImport(choice.setSchema ? res.view : null);
			toast(`Imported ${plural(res.added, "example")}` + (res.skipped ? `; skipped ${res.skipped.toLocaleString()} without a text or label` : ""));
		} catch (e) {
			busy.close();
			toastError(e);
		}
	};
	dialog((close) => h("form", { method: "dialog", onsubmit: (e) => { e.preventDefault(); run(close); } },
		h("div", { class: "dialog-head" }, h("div", {}, h("h2", {}, "Import from Hugging Face"),
			h("p", {}, "Labeled classification datasets such as mteb/banking77 become examples."))),
		h("div", { class: "dialog-body stack" },
			h("div", { class: "row" }, h("div", { class: "grow" }, field("Dataset", id)), button("Look up", { kind: "secondary", onclick: lookUp })),
			body),
		h("div", { class: "dialog-foot" }, button("Cancel", { kind: "ghost", onclick: close }), button("Import", { kind: "primary", type: "submit" })),
	));
	id.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); lookUp(); } });
	if (id.value) lookUp();
}

async function watchSynthesis(busy) {
	for (;;) {
		await new Promise((r) => setTimeout(r, 1500));
		const st = await api(`/projects/${S.pid}/synthesize`);
		if (st.total) busy.progress(st.done / st.total);
		busy.text(`${st.done} of ${st.total || "…"} answers done, ${plural(st.added || 0, "example")} added`);
		if (!st.running) return st;
	}
}

export function synthesizeDialog() {
	const perValue = h("input", { type: "number", min: 1, max: 50, value: 10 });
	dialog((close) => h("form", { method: "dialog", onsubmit: async (e) => {
		e.preventDefault();
		close();
		try {
			await saver.flush();
			await api(`/projects/${S.pid}/synthesize`, { method: "POST", body: { per_value: Number(perValue.value) || 10 } });
		} catch (err) {
			toastError(err);
			return;
		}
		const busy = progressDialog("Writing examples", "Asking the generator…");
		try {
			const st = await watchSynthesis(busy);
			busy.close();
			await afterImport(null);
			if (st.error) toast(`Stopped: ${st.error}`, true);
			else toast(`Added ${plural(st.added, "synthetic example")}. Their ids start with syn-; check a few before training.`);
		} catch (err) {
			busy.close();
			toastError(err);
		}
	} },
		h("div", { class: "dialog-head" }, h("div", {}, h("h2", {}, "Generate examples with an LLM"),
			h("p", {}, "A stronger model writes inputs for every value of each option list and yes/no field, including hard cases that look like another value. Every answer is checked against the output schema."))),
		h("div", { class: "dialog-body stack" },
			field("Examples per value", perValue),
			h("p", { class: "faint small" }, "Uses the generator in Settings (any OpenAI-compatible endpoint; Ollama by default)."),
		),
		h("div", { class: "dialog-foot" }, button("Cancel", { kind: "ghost", onclick: close }), button("Generate", { kind: "primary", type: "submit" })),
	));
}
