import { imageUrl } from "./api.js";
import { button, h, icon, mount } from "./dom.js";
import { deleteExamples, visibleExamples } from "./example-ops.js";
import { toastError } from "./feedback.js";
import { pickImages, uploadPrepared } from "./intake.js";
import { labelText } from "./format.js";
import { jsonBlock } from "./json-view.js";
import { outputForm } from "./output-form.js";
import { scheduleExamplesSave } from "./project.js";
import { blankOutput, labelFields, labelValues } from "./schema-model.js";
import { S, app, exampleErrors, schemaTree } from "./state.js";

const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const typing = (el) => el && (el.matches("textarea, select") || (el.matches("input") && !["checkbox", "radio"].includes(el.type)));

// Full-size view of one example with its answer; also a keyboard-driven labeling mode.
export function openReview(id) {
	let list = visibleExamples();
	let index = Math.max(0, list.findIndex((e) => e.id === id));
	const dlg = h("dialog", { class: "review", "aria-label": "Review example" });

	const quickField = () => {
		const parsed = schemaTree();
		return parsed.ok ? labelFields(parsed.root)[0] || null : null;
	};

	function go(delta) {
		list = visibleExamples();
		if (!list.length) { dlg.close(); return; }
		index = Math.min(list.length - 1, Math.max(0, index + delta));
		render();
	}

	function label(value) {
		const ex = list[index];
		const field = quickField();
		if (!ex || !field) return;
		if (!isObj(ex.output)) ex.output = blankOutput(S.config.schema);
		ex.output[field.name] = value;
		scheduleExamplesSave();
		if (index < list.length - 1) go(1);
		else render();
	}

	function replaceImage(ex) {
		pickImages(async ([file]) => {
			if (!file) return;
			try {
				ex.image = await uploadPrepared(file);
				scheduleExamplesSave();
				render();
			} catch (e) {
				toastError(e);
			}
		});
	}

	function render() {
		const ex = list[index];
		if (!ex) { dlg.close(); return; }
		const parsed = schemaTree();
		const errors = exampleErrors(ex.id);
		const field = quickField();
		const text = h("textarea", { rows: 3, value: ex.input, placeholder: "Optional text sent with the image", "data-focus": `${ex.id}:input` });
		text.addEventListener("input", () => { ex.input = text.value; scheduleExamplesSave(); });
		const answer = parsed.ok && isObj(ex.output)
			? outputForm(parsed.root, ex.output, () => scheduleExamplesSave(), { keyPrefix: `review-${ex.id}` })
			: jsonBlock(ex.output ?? null);
		mount(dlg,
			h("div", { class: "review-media" },
				ex.image ? h("img", { src: imageUrl(S.pid, ex.image), alt: ex.input || "Example image" }) : h("div", { class: "review-missing" }, icon("image"), "This example has no image"),
				h("button", { type: "button", class: "review-nav prev", "aria-label": "Previous example", disabled: index === 0, onclick: () => go(-1) }, icon("chevron-left")),
				h("button", { type: "button", class: "review-nav next", "aria-label": "Next example", disabled: index >= list.length - 1, onclick: () => go(1) }, icon("chevron-right")),
			),
			h("div", { class: "review-side" },
				h("div", { class: "spread" },
					h("span", { class: "muted num" }, `${index + 1} of ${list.length}`),
					button(null, { icon: "x", kind: "ghost", title: "Close", onclick: () => dlg.close() }),
				),
				h("div", { class: "field" }, h("span", { class: "field-label" }, "Answer"), answer),
				errors.length ? h("div", { class: "callout bad small" }, icon("alert-circle"), h("div", {}, errors.map((e) => h("div", {}, e)))) : null,
				h("label", { class: "field" }, h("span", { class: "field-label" }, "Text", h("span", { class: "faint" }, "optional")), text),
				h("div", { class: "row" },
					button(ex.image ? "Replace image" : "Add image", { icon: "image", kind: "secondary", size: "sm", onclick: () => replaceImage(ex) }),
					button("Delete", { icon: "trash", kind: "danger-quiet", size: "sm", onclick: () => { deleteExamples([ex.id]); go(0); } }),
				),
				h("div", { class: "grow" }),
				h("dl", { class: "review-keys small" },
					field ? [
						h("dt", {}, labelValues(field).slice(0, 9).map((v, i) => h("kbd", { title: labelText(v) }, String(i + 1)))),
						h("dd", {}, `${labelValues(field).slice(0, 9).map(labelText).join(" / ")}, then next`),
					] : null,
					h("dt", {}, h("kbd", {}, "←"), h("kbd", {}, "→")), h("dd", {}, "Previous / next"),
					h("dt", {}, h("kbd", {}, "Del")), h("dd", {}, "Delete (you can undo)"),
				),
			),
		);
	}

	dlg.addEventListener("keydown", (e) => {
		if (typing(e.target) || e.ctrlKey || e.metaKey || e.altKey) return;
		if (e.key === "ArrowLeft") { e.preventDefault(); go(-1); }
		else if (e.key === "ArrowRight") { e.preventDefault(); go(1); }
		else if (e.key === "Delete") { e.preventDefault(); const ex = list[index]; if (ex) { deleteExamples([ex.id]); go(0); } }
		else if (/^[1-9]$/.test(e.key)) {
			const field = quickField();
			const values = field ? labelValues(field) : [];
			const v = values[Number(e.key) - 1];
			if (v !== undefined) { e.preventDefault(); label(v); }
		}
	});
	dlg.addEventListener("close", () => {
		dlg.remove();
		app.renderExamples({ keepScroll: true });
	});
	render();
	document.body.append(dlg);
	dlg.showModal();
	dlg.focus();
}
