import { uploadFile } from "./api.js";
import { $, $$, button, callout, emptyState, h, icon, menu, mount } from "./dom.js";
import { addExamples, deleteExamples, reloadExamples } from "./example-ops.js";
import { progressDialog, saver, toast, toastError } from "./feedback.js";
import { bytes, labelText, plural } from "./format.js";
import { gallery, selectAllToggle, selectionBar, toggleSelect } from "./gallery.js";
import { hubImportDialog, synthesizeDialog } from "./assist.js";
import { healthPanel } from "./health.js";
import { openIntake, pickImages } from "./intake.js";
import { outputForm } from "./output-form.js";
import { loadPage, pageSize, saveExample } from "./project.js";
import { blankOutput } from "./schema-model.js";
import { S, app, exampleErrors, isImageProject, schemaTree } from "./state.js";

const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const CHARS_PER_TOKEN = 3.5;
const IMPORT_MODES = { append: "Add to these", update: "Update matching ids, add the rest", replace: "Replace all" };

function autoGrow(textarea) {
	const fit = () => {
		textarea.style.height = "auto";
		textarea.style.height = `${Math.min(textarea.scrollHeight + 2, 420)}px`;
	};
	textarea.addEventListener("input", fit);
	requestAnimationFrame(fit);
	return textarea;
}

// ---------- text examples ----------

function readTextFile(ex) {
	const input = h("input", { type: "file" });
	input.addEventListener("change", async () => {
		const file = input.files[0];
		if (!file) return;
		if (file.size > 512 * 1024) toast(`${file.name} is ${bytes(file.size)}. Long inputs need a lot of memory to train on.`, true);
		ex.input = await file.text();
		saveExample(ex);
		renderList();
	});
	input.click();
}

function jsonAnswer(ex) {
	const area = h("textarea", { class: "code", rows: 3, spellcheck: "false", value: JSON.stringify(ex.output, null, 2), "data-focus": `${ex.id}:json`, "aria-label": "Answer as JSON" });
	area.addEventListener("input", () => {
		try {
			ex.output = JSON.parse(area.value);
			S.badJson.delete(ex.id);
			saveExample(ex);
		} catch (err) {
			S.badJson.set(ex.id, `The answer isn't valid JSON: ${err.message}`);
		}
		updateErrors();
	});
	return autoGrow(area);
}

const tokens = (text) => Math.ceil(text.length / CHARS_PER_TOKEN);

function exampleRow(ex, index) {
	const parsed = schemaTree();
	const formOk = parsed.ok && isObj(ex.output);
	const asJson = !formOk || S.jsonEditing.has(ex.id);
	const count = h("span", { class: "faint small num", title: "Rough token count" }, `≈${tokens(ex.input).toLocaleString()} tokens`);
	const input = autoGrow(h("textarea", { rows: 2, value: ex.input, placeholder: "The input text…", "data-focus": `${ex.id}:input`, "aria-label": "Input" }));
	input.addEventListener("input", () => {
		ex.input = input.value;
		count.textContent = `≈${tokens(ex.input).toLocaleString()} tokens`;
		saveExample(ex);
	});
	const flip = () => {
		if (asJson) S.jsonEditing.delete(ex.id);
		else S.jsonEditing.add(ex.id);
		renderList();
	};
	const n = S.query.offset + index + 1;
	const pick = h("input", { type: "checkbox", checked: S.selection.has(ex.id), title: "Select (Shift-click for a range)", "aria-label": `Select example ${n}` });
	pick.addEventListener("click", (e) => toggleSelect(ex, S.examples, e.shiftKey));
	return h("article", { class: `example-row card${S.selection.has(ex.id) ? " selected" : ""}`, dataset: { id: ex.id } },
		h("div", { class: "ex-col" },
			h("div", { class: "ex-label" }, pick, h("span", {}, "Input"), h("span", { class: "faint num" }, `#${n.toLocaleString()}`), count, h("span", { class: "grow" }),
				button("From file", { icon: "upload", kind: "ghost", size: "sm", title: "Replace the input with a text file's contents", onclick: () => readTextFile(ex) })),
			input,
		),
		h("div", { class: "ex-col" },
			h("div", { class: "ex-label" }, h("span", {}, "Answer"), h("span", { class: "grow" }),
				formOk ? button(asJson ? "Form" : "JSON", { icon: asJson ? "list" : "braces", kind: "ghost", size: "sm", title: asJson ? "Edit with the form" : "Edit the raw JSON", onclick: flip }) : null),
			asJson ? jsonAnswer(ex) : outputForm(parsed.root, ex.output, () => saveExample(ex), { keyPrefix: ex.id }),
		),
		button(null, { icon: "trash", kind: "ghost", size: "sm", title: "Delete example", cls: "ex-delete", onclick: () => deleteExamples({ ids: [ex.id] }) }),
		h("div", { class: "ex-errors" }),
	);
}

function updateErrors() {
	for (const el of $$("#tab-examples [data-id]")) {
		const errors = exampleErrors(el.dataset.id);
		el.classList.toggle("bad", errors.length > 0);
		const box = $(".ex-errors", el);
		if (box) mount(box, errors.map((e) => h("div", {}, icon("alert-circle"), e)));
		for (const flag of $$(".gflag, .gerror", el)) {
			flag.hidden = !errors.length;
			flag.title = errors.join("\n");
		}
		const line = $(".gerror", el);
		if (line) line.textContent = errors[0] || "";
	}
}

// ---------- adding, importing, exporting ----------

async function addTextExample() {
	try {
		await addExamples([{ input: "", output: blankOutput(S.config.schema) }]);
		S.query = { filter: "all", search: "", label: null, offset: 0 };
		await reloadExamples();
		const first = $("#tab-examples .example-row textarea");
		if (first) first.focus();
	} catch (e) {
		toastError(e);
	}
}

function importFile(mode) {
	const input = h("input", { type: "file", accept: ".jsonl,.json,.csv,.tsv,.txt" });
	input.addEventListener("change", async () => {
		const file = input.files[0];
		if (!file) return;
		await saver.flush();
		const busy = progressDialog(`Importing ${file.name}`, `Uploading ${bytes(file.size)}…`);
		try {
			const res = await uploadFile(`/projects/${S.pid}/import?${new URLSearchParams({ filename: file.name, mode })}`, file, (share) => {
				busy.progress(share);
				if (share >= 1) busy.text("Reading the examples…");
			});
			S.importWarnings = res.warning_count ? { count: res.warning_count, lines: res.warnings } : null;
			S.query = { filter: "all", search: "", label: null, offset: 0 };
			await reloadExamples();
			toast(`Imported ${plural(res.added, "new example")}` + (res.updated ? `, updated ${res.updated.toLocaleString()}` : "") + (res.warning_count ? `; skipped ${res.warning_count.toLocaleString()}` : ""));
		} catch (e) {
			toastError(e);
		} finally {
			busy.close();
		}
	});
	input.click();
}

function downloadJsonl() {
	h("a", { href: `/api/projects/${encodeURIComponent(S.pid)}/examples/export`, download: `${S.pid}-examples.jsonl` }).click();
}

// ---------- layout ----------

function header() {
	const image = isImageProject();
	return h("div", { class: "spread" },
		h("div", {},
			h("h2", { class: "section-title" }, "Examples"),
			h("p", { class: "muted small" }, image ? "Images paired with the exact answer you expect. Drop or paste images anywhere on this page." : "Inputs paired with the exact answer you expect."),
		),
		h("div", { class: "row" },
			menu(button("Import / export", { icon: "folder", kind: "secondary" }), [
				...Object.entries(IMPORT_MODES).map(([mode, text]) => ({ label: `Import a file: ${text.toLowerCase()}`, icon: "upload", onclick: () => importFile(mode) })),
				image ? null : { label: "Import from Hugging Face…", icon: "database", onclick: hubImportDialog },
				image ? null : { label: "Generate examples with an LLM…", icon: "layers", onclick: synthesizeDialog },
				"-",
				{ label: "Download as JSONL", icon: "download", disabled: !S.counts.all, onclick: downloadJsonl },
			]),
			image
				? button("Add images", { icon: "image", kind: "primary", onclick: () => pickImages(openIntake) })
				: button("Add example", { icon: "plus", kind: "primary", onclick: addTextExample }),
		),
	);
}

async function setQuery(changes) {
	Object.assign(S.query, { offset: 0 }, changes);
	S.selection.clear();
	S.selectAll = false;
	try {
		await loadPage({ stats: false });
	} catch (e) {
		toastError(e);
	}
	renderList();
}

function filterBar() {
	const q = S.query;
	const chip = (key, text, n) => h("button", { type: "button", class: "chip", "aria-pressed": String(q.filter === key), onclick: () => setQuery({ filter: key }) },
		text, h("span", { class: "count" }, n.toLocaleString()));
	const search = h("input", { type: "search", value: q.search, placeholder: "Search inputs and answers", "data-focus": "ex-search", "aria-label": "Search examples" });
	let timer;
	search.addEventListener("input", () => {
		clearTimeout(timer);
		timer = setTimeout(() => setQuery({ search: search.value }), 250);
	});
	return h("div", { class: "filter-bar" },
		h("div", { class: "search-box" }, icon("search"), search),
		h("div", { class: "row" },
			chip("all", "All", S.counts.all),
			S.counts.problems || q.filter === "problems" ? chip("problems", "To fix", S.counts.problems) : null,
			S.counts.duplicates || q.filter === "duplicates" ? chip("duplicates", "Repeated", S.counts.duplicates) : null,
			q.label ? h("span", { class: "chip", "aria-pressed": "true" }, icon("filter"), `${q.label.field}: ${labelText(q.label.value)}`,
				h("button", { type: "button", class: "x", "aria-label": "Clear the value filter", onclick: () => setQuery({ label: null }) }, icon("x"))) : null,
		),
		h("span", { class: "grow" }),
		S.examples.length ? selectAllToggle(S.examples) : null,
	);
}

function pager() {
	const size = pageSize();
	if (S.total <= size) return null;
	const go = async (offset) => {
		await saver.flush();
		S.query.offset = offset;
		await loadPage({ stats: false });
		renderList();
		$("#tab-examples").scrollIntoView({ block: "start" });
	};
	const start = S.query.offset;
	return h("div", { class: "pager" },
		button(null, { icon: "chevron-left", kind: "secondary", size: "sm", title: "Previous page", disabled: start === 0, onclick: () => go(Math.max(0, start - size)) }),
		h("span", { class: "muted num" }, `${(start + 1).toLocaleString()}–${Math.min(S.total, start + size).toLocaleString()} of ${S.total.toLocaleString()}`),
		button(null, { icon: "chevron-right", kind: "secondary", size: "sm", title: "Next page", disabled: start + size >= S.total, onclick: () => go(start + size) }),
	);
}

function emptyView() {
	if (S.counts.all) {
		return emptyState("search", "Nothing matches", "No examples match the current filter or search.",
			button("Show all", { kind: "secondary", onclick: () => setQuery({ filter: "all", label: null, search: "" }) }));
	}
	if (isImageProject()) {
		return h("button", { type: "button", class: "dropzone big", onclick: () => pickImages(openIntake) },
			icon("upload"), h("b", {}, "Drop images here, paste them, or click to choose"),
			h("span", { class: "muted" }, "Add a batch of pictures that need the same answer, then the next batch. Aim for at least 20 per label."));
	}
	return emptyState("text", "No examples yet", "Add inputs with the answer you expect, or import a JSONL or CSV file of any size. Aim for at least 20; 100+ covering every label is better.",
		button("Add example", { icon: "plus", kind: "primary", onclick: addTextExample }),
		button("Import a file", { icon: "upload", kind: "secondary", onclick: () => importFile("append") }));
}

function renderList() {
	const rows = S.examples;
	mount($("#ex-filters"), S.counts.all ? filterBar() : null);
	mount($("#ex-list"), !rows.length ? emptyView() : isImageProject() ? gallery(rows) : h("div", { class: "example-list" }, rows.map(exampleRow)));
	mount($("#ex-pager"), pager());
	mount($("#ex-selection"), selectionBar());
	updateErrors();
}

function renderHealth() {
	mount($("#ex-health"), S.counts.all ? healthPanel({
		onFilter: (label) => setQuery({ label }),
		onShow: (filter) => setQuery({ filter }),
	}) : null);
}


export function renderExamples({ keepScroll = false } = {}) {
	const y = window.scrollY;
	const warnings = S.importWarnings;
	mount($("#tab-examples"),
		h("div", { class: "stack-lg" },
			header(),
			warnings ? callout("warn", "alert",
				h("b", {}, `${plural(warnings.count, "line")} couldn't be imported`),
				h("ul", {}, warnings.lines.slice(0, 8).map((w) => h("li", {}, w))),
				warnings.count > 8 ? h("p", { class: "small" }, `…and ${(warnings.count - 8).toLocaleString()} more.`) : null,
				button("Dismiss", { kind: "ghost", size: "sm", onclick: () => { S.importWarnings = null; renderExamples({ keepScroll: true }); } })) : null,
			h("div", { id: "ex-health" }),
			h("div", { class: "stack" }, h("div", { id: "ex-filters" }), h("div", { id: "ex-list" }), h("div", { id: "ex-pager" })),
		),
		h("div", { id: "ex-selection", class: "selection-slot" }),
	);
	renderHealth();
	renderList();
	if (keepScroll) window.scrollTo(0, y);
}

app.renderExamples = (opts) => { if (S.tab === "examples") renderExamples(opts); };
app.examplesChecked = () => {
	if (S.tab !== "examples") return;
	updateErrors();
	renderHealth();
	const bar = $("#ex-filters");
	if (bar && S.counts.all && !bar.contains(document.activeElement)) mount(bar, filterBar());
};
document.addEventListener("examples-changed", () => reloadExamples().catch(toastError));
