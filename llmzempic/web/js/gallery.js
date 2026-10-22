import { imageUrl } from "./api.js";
import { button, h, icon, mount } from "./dom.js";
import { deleteExamples, selectionScope, setAnswer } from "./example-ops.js";
import { clip, plural } from "./format.js";
import { jsonBlock } from "./json-view.js";
import { outputForm } from "./output-form.js";
import { saveExample } from "./project.js";
import { openReview } from "./review.js";
import { blankOutput, labelFields } from "./schema-model.js";
import { S, app, exampleErrors, schemaTree } from "./state.js";

const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
let lastClicked = null;

export function toggleSelect(ex, page, shift) {
	const ids = page.map((e) => e.id);
	if (shift && lastClicked && ids.includes(lastClicked)) {
		const [a, b] = [ids.indexOf(lastClicked), ids.indexOf(ex.id)].sort((x, y) => x - y);
		const on = !S.selection.has(ex.id);
		for (const id of ids.slice(a, b + 1)) on ? S.selection.add(id) : S.selection.delete(id);
	} else if (S.selection.has(ex.id)) {
		S.selection.delete(ex.id);
	} else {
		S.selection.add(ex.id);
	}
	S.selectAll = false;
	lastClicked = ex.id;
	app.renderExamples({ keepScroll: true });
}

function cardBody(ex) {
	const parsed = schemaTree();
	if (parsed.ok && isObj(ex.output) && labelFields(parsed.root).length) {
		return outputForm(parsed.root, ex.output, () => saveExample(ex), { labelsOnly: true, keyPrefix: ex.id });
	}
	return h("button", { type: "button", class: "gcard-json", title: "Edit answer", onclick: () => openReview(ex.id) },
		jsonBlock(ex.output ?? null, "tiny"));
}

function card(ex, page) {
	const errors = exampleErrors(ex.id);
	const selected = S.selection.has(ex.id);
	return h("article", { class: `gcard${selected ? " selected" : ""}${errors.length ? " bad" : ""}`, dataset: { id: ex.id } },
		h("div", { class: "gthumb" },
			ex.image
				? h("button", { type: "button", class: "gthumb-open", title: "Open", onclick: () => openReview(ex.id) },
					h("img", { src: imageUrl(S.pid, ex.image), alt: ex.input ? clip(ex.input, 80) : "Example image", loading: "lazy", decoding: "async" }))
				: h("button", { type: "button", class: "gthumb-open missing", onclick: () => openReview(ex.id) }, icon("image"), "No image"),
			h("label", { class: "gselect", title: "Select (Shift-click for a range)", onclick: (e) => { e.preventDefault(); toggleSelect(ex, page, e.shiftKey); } },
				h("input", { type: "checkbox", checked: selected, tabindex: -1 }), h("span", { class: "sr-only" }, "Select")),
			h("span", { class: "gflag", hidden: !errors.length, title: errors.join("\n") }, icon("alert")),
		),
		h("div", { class: "gbody" },
			cardBody(ex),
			ex.input ? h("p", { class: "gtext faint small ellipsis", title: ex.input }, ex.input) : null,
			h("p", { class: "gerror small", hidden: !errors.length, title: errors.join("\n") }, errors[0] || ""),
		),
	);
}

export function gallery(page) {
	return h("div", { class: "gallery" }, page.map((ex) => card(ex, page)));
}

export function selectAllToggle(page) {
	const all = page.length > 0 && page.every((e) => S.selection.has(e.id));
	const box = h("input", { type: "checkbox", checked: all });
	box.indeterminate = !all && page.some((e) => S.selection.has(e.id));
	box.addEventListener("change", () => {
		for (const e of page) box.checked ? S.selection.add(e.id) : S.selection.delete(e.id);
		S.selectAll = false;
		app.renderExamples({ keepScroll: true });
	});
	return h("label", { class: "check small muted" }, box, "Select page");
}

function bulkAnswerDialog(scope, count) {
	const parsed = schemaTree();
	const first = S.examples.find((e) => S.selection.has(e.id));
	const answer = structuredClone(isObj(first && first.output) ? first.output : blankOutput(S.config.schema));
	const dlg = h("dialog", {});
	const close = () => { dlg.close(); dlg.remove(); };
	mount(dlg,
		h("div", { class: "dialog-head" }, h("div", {},
			h("h2", {}, `Set the answer for ${plural(count, "example")}`),
			h("p", {}, "Every selected example gets exactly this answer."))),
		h("div", { class: "dialog-body" }, parsed.ok ? outputForm(parsed.root, answer, () => {}, { keyPrefix: "bulk" }) : h("p", { class: "faint" }, "Edit answers one at a time for this schema.")),
		h("div", { class: "dialog-foot" },
			button("Cancel", { kind: "ghost", onclick: close }),
			button("Apply", { kind: "primary", disabled: !parsed.ok, onclick: () => { close(); setAnswer(scope, answer); S.lastLabel = structuredClone(answer); } }),
		),
	);
	dlg.addEventListener("close", () => dlg.remove());
	document.body.append(dlg);
	dlg.showModal();
}

export function selectionBar() {
	if (S.busy) return h("div", { class: "selection-bar", role: "status" }, h("span", { class: "spinner" }), h("b", {}, S.busy));
	const ids = S.examples.filter((e) => S.selection.has(e.id)).map((e) => e.id);
	if (!ids.length) return null;
	const count = S.selectAll ? S.total : ids.length;
	const wholePage = ids.length === S.examples.length && S.total > ids.length;
	return h("div", { class: "selection-bar", role: "region", "aria-label": "Selection" },
		h("b", {}, S.selectAll ? `All ${count.toLocaleString()} matching` : plural(count, "selected", "selected")),
		wholePage && !S.selectAll ? h("button", { type: "button", class: "link-btn small", onclick: () => { S.selectAll = true; app.renderExamples({ keepScroll: true }); } },
			`Select all ${S.total.toLocaleString()} matching`) : null,
		button("Set answer…", { icon: "pencil", kind: "secondary", size: "sm", onclick: () => bulkAnswerDialog(selectionScope(), count) }),
		button("Delete", { icon: "trash", kind: "danger-quiet", size: "sm", onclick: () => deleteExamples(selectionScope()) }),
		button("Clear", { kind: "ghost", size: "sm", onclick: () => { S.selection.clear(); S.selectAll = false; app.renderExamples({ keepScroll: true }); } }),
	);
}
