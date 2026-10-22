import { api } from "./api.js";
import { $, $$, button, emptyState, h, icon, menu, mount } from "./dom.js";
import { deleteExamples, filterCounts, visibleExamples } from "./example-ops.js";
import { saver, toast, toastError } from "./feedback.js";
import { bytes, labelText, plural, randomId } from "./format.js";
import { gallery, selectAllToggle, selectionBar } from "./gallery.js";
import { healthPanel } from "./health.js";
import { openIntake, pickImages } from "./intake.js";
import { outputForm } from "./output-form.js";
import { scheduleExamplesSave } from "./project.js";
import { blankOutput } from "./schema-model.js";
import { S, app, exampleErrors, isImageProject, schemaTree, setExamplesData } from "./state.js";

const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const pageSize = () => (isImageProject() ? 48 : 20);

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
		scheduleExamplesSave();
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
		} catch (err) {
			S.badJson.set(ex.id, `The answer isn't valid JSON: ${err.message}`);
		}
		updateErrors();
		scheduleExamplesSave();
	});
	return autoGrow(area);
}

function exampleRow(ex) {
	const parsed = schemaTree();
	const formOk = parsed.ok && isObj(ex.output);
	const asJson = !formOk || S.jsonEditing.has(ex.id);
	const input = autoGrow(h("textarea", { rows: 2, value: ex.input, placeholder: "The input text…", "data-focus": `${ex.id}:input`, "aria-label": "Input" }));
	input.addEventListener("input", () => { ex.input = input.value; scheduleExamplesSave(); });
	const flip = () => {
		if (asJson) S.jsonEditing.delete(ex.id);
		else S.jsonEditing.add(ex.id);
		renderList();
	};
	return h("article", { class: "example-row card", dataset: { id: ex.id } },
		h("div", { class: "ex-col" },
			h("div", { class: "ex-label" }, h("span", {}, "Input"), h("span", { class: "faint num" }, `#${S.examples.indexOf(ex) + 1}`), h("span", { class: "grow" }),
				button("From file", { icon: "upload", kind: "ghost", size: "sm", title: "Replace the input with a text file's contents", onclick: () => readTextFile(ex) })),
			input,
		),
		h("div", { class: "ex-col" },
			h("div", { class: "ex-label" }, h("span", {}, "Answer"), h("span", { class: "grow" }),
				formOk ? button(asJson ? "Form" : "JSON", { icon: asJson ? "list" : "braces", kind: "ghost", size: "sm", title: asJson ? "Edit with the form" : "Edit the raw JSON", onclick: flip }) : null),
			asJson ? jsonAnswer(ex) : outputForm(parsed.root, ex.output, () => scheduleExamplesSave(), { keyPrefix: ex.id }),
		),
		button(null, { icon: "trash", kind: "ghost", size: "sm", title: "Delete example", cls: "ex-delete", onclick: () => deleteExamples([ex.id]) }),
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

// ---------- adding & importing ----------

function addTextExample() {
	S.examples.unshift({ id: randomId(), input: "", output: blankOutput(S.config.schema) });
	Object.assign(S, { page: 0, filter: "all", labelFilter: null, search: "" });
	scheduleExamplesSave();
	renderExamples();
	const first = $("#tab-examples .example-row textarea");
	if (first) first.focus();
}

function importFile(mode) {
	const input = h("input", { type: "file", accept: ".jsonl,.json,.csv,.tsv,.txt" });
	input.addEventListener("change", async () => {
		const file = input.files[0];
		if (!file) return;
		try {
			await saver.flush();
			const data = await api(`/projects/${S.pid}/import`, { method: "POST", body: { text: await file.text(), filename: file.name, mode } });
			setExamplesData(data);
			S.page = 0;
			S.importWarnings = data.warnings;
			renderExamples();
			app.renderStepper();
			toast(`Imported ${plural(data.imported, "example")}` + (data.warnings.length ? `; skipped ${data.warnings.length}` : ""));
		} catch (e) {
			toastError(e);
		}
	});
	input.click();
}

function downloadJsonl() {
	const text = S.examples.map((e) => JSON.stringify(e.image ? { input: e.input, image: e.image, output: e.output } : { input: e.input, output: e.output })).join("\n") + "\n";
	const a = h("a", { href: URL.createObjectURL(new Blob([text], { type: "application/jsonl" })), download: `${S.pid}-examples.jsonl` });
	a.click();
	setTimeout(() => URL.revokeObjectURL(a.href), 1000);
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
				{ label: "Import a file (add to these)", icon: "upload", onclick: () => importFile("append") },
				{ label: "Import a file (replace all)", icon: "upload", onclick: () => importFile("replace") },
				"-",
				{ label: "Download as JSONL", icon: "download", disabled: !S.examples.length, onclick: downloadJsonl },
			]),
			image
				? button("Add images", { icon: "image", kind: "primary", onclick: () => pickImages(openIntake) })
				: button("Add example", { icon: "plus", kind: "primary", onclick: addTextExample }),
		),
	);
}

function setFilter(changes) {
	Object.assign(S, { page: 0 }, changes);
	renderExamples();
}

function filterBar(page) {
	const counts = filterCounts();
	const chip = (key, text, n) => h("button", { type: "button", class: "chip", "aria-pressed": String(S.filter === key), onclick: () => setFilter({ filter: key }) },
		text, h("span", { class: "count" }, n.toLocaleString()));
	const search = h("input", { type: "search", value: S.search, placeholder: "Search inputs and answers", "data-focus": "ex-search", "aria-label": "Search examples" });
	let timer;
	search.addEventListener("input", () => {
		clearTimeout(timer);
		timer = setTimeout(() => { S.search = search.value; S.page = 0; renderList(); }, 200);
	});
	return h("div", { class: "filter-bar" },
		h("div", { class: "search-box" }, icon("search"), search),
		h("div", { class: "row" },
			chip("all", "All", counts.all),
			counts.problems || S.filter === "problems" ? chip("problems", "To fix", counts.problems) : null,
			counts.duplicates || S.filter === "duplicates" ? chip("duplicates", "Repeated", counts.duplicates) : null,
			S.labelFilter ? h("span", { class: "chip", "aria-pressed": "true" }, icon("filter"), `${S.labelFilter.field}: ${labelText(S.labelFilter.value)}`,
				h("button", { type: "button", class: "x", "aria-label": "Clear label filter", onclick: () => setFilter({ labelFilter: null }) }, icon("x"))) : null,
		),
		h("span", { class: "grow" }),
		isImageProject() && page.length ? selectAllToggle(page) : null,
	);
}

function pager(total) {
	const size = pageSize();
	const pages = Math.max(1, Math.ceil(total / size));
	if (pages <= 1) return null;
	const go = (p) => { S.page = p; renderList(); $("#tab-examples").scrollIntoView({ block: "start" }); };
	const start = S.page * size;
	return h("div", { class: "pager" },
		button(null, { icon: "chevron-left", kind: "secondary", size: "sm", title: "Previous page", disabled: S.page === 0, onclick: () => go(S.page - 1) }),
		h("span", { class: "muted num" }, `${start + 1}–${Math.min(total, start + size)} of ${total.toLocaleString()}`),
		button(null, { icon: "chevron-right", kind: "secondary", size: "sm", title: "Next page", disabled: S.page >= pages - 1, onclick: () => go(S.page + 1) }),
	);
}

function emptyView() {
	if (S.examples.length) {
		return emptyState("search", "Nothing matches", "No examples match the current filter or search.",
			button("Show all", { kind: "secondary", onclick: () => setFilter({ filter: "all", labelFilter: null, search: "" }) }));
	}
	if (isImageProject()) {
		const zone = h("button", { type: "button", class: "dropzone big", onclick: () => pickImages(openIntake) },
			icon("upload"), h("b", {}, "Drop images here, paste them, or click to choose"),
			h("span", { class: "muted" }, "Add a batch of pictures that need the same answer, then the next batch. Aim for at least 20 per label."));
		return zone;
	}
	return emptyState("text", "No examples yet", "Add inputs with the answer you expect, or import a JSONL or CSV file. Aim for at least 20; 100+ covering every label is better.",
		button("Add example", { icon: "plus", kind: "primary", onclick: addTextExample }),
		button("Import a file", { icon: "upload", kind: "secondary", onclick: () => importFile("append") }));
}

function renderList() {
	const list = visibleExamples();
	const size = pageSize();
	S.page = Math.min(S.page, Math.max(0, Math.ceil(list.length / size) - 1));
	const page = list.slice(S.page * size, S.page * size + size);
	mount($("#ex-filters"), S.examples.length ? filterBar(page) : null);
	mount($("#ex-list"), !page.length ? emptyView() : isImageProject() ? gallery(page) : h("div", { class: "example-list" }, page.map(exampleRow)));
	mount($("#ex-pager"), pager(list.length));
	mount($("#ex-selection"), isImageProject() ? selectionBar() : null);
	updateErrors();
}

function renderHealth() {
	mount($("#ex-health"), S.examples.length ? healthPanel({
		onFilter: (labelFilter) => setFilter({ labelFilter }),
		onShow: (filter) => setFilter({ filter }),
	}) : null);
}

export function renderExamples({ keepScroll = false } = {}) {
	const y = window.scrollY;
	const pane = $("#tab-examples");
	const warnings = S.importWarnings || [];
	mount(pane,
		h("div", { class: "stack-lg" },
			header(),
			warnings.length ? h("div", { class: "callout warn" }, icon("alert"), h("div", { class: "grow" },
				h("b", {}, `${plural(warnings.length, "record")} skipped during import`),
				h("ul", {}, warnings.slice(0, 8).map((w) => h("li", {}, w))),
				button("Dismiss", { kind: "ghost", size: "sm", onclick: () => { S.importWarnings = []; renderExamples({ keepScroll: true }); } }))) : null,
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
	const page = visibleExamples();
	const bar = $("#ex-filters");
	if (bar && S.examples.length && !bar.contains(document.activeElement)) mount(bar, filterBar(page.slice(S.page * pageSize(), S.page * pageSize() + pageSize())));
};
document.addEventListener("examples-changed", () => { app.renderExamples(); app.renderStepper(); });
