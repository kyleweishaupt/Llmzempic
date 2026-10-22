import { $, $$, button, h, icon, menu, mount, preserveFocus, toggleSwitch } from "./dom.js";
import { ITEM_KINDS, KINDS, changeKind, newNode, problems, toSchema } from "./schema-model.js";

// Visual editor for a schema tree (see schema-model.js). Calls onChange(schema) after every edit
// that leaves the tree valid; invalid edits are highlighted and not emitted.
export function schemaEditor(root, onChange) {
	const el = h("div", { class: "schema-editor" });
	let issues = problems(root);
	let drag = null;

	function commit(structural) {
		issues = problems(root);
		if (structural) preserveFocus(render);
		else showIssues();
		if (!issues.errors.size) onChange(toSchema(root));
	}

	function showIssues() {
		for (const row of $$(".field-row", el)) {
			const id = row.dataset.id;
			const msg = issues.errors.get(id) || issues.warnings.get(id) || "";
			row.classList.toggle("has-error", issues.errors.has(id));
			const line = $(`:scope > .field-body > .field-msg`, row);
			line.textContent = msg;
			line.className = `field-msg ${issues.errors.has(id) ? "error" : "warning"}`;
			line.hidden = !msg;
		}
		const banner = $(":scope > .schema-banner", el);
		if (banner) banner.hidden = !issues.errors.size;
	}

	function move(list, from, to) {
		const [node] = list.splice(from, 1);
		list.splice(to, 0, node);
		commit(true);
	}

	function optionsEditor(node) {
		const input = h("input", {
			type: "text", class: "chip-input", placeholder: node.options.length ? "Add option" : "Type an option and press Enter",
			"data-focus": `${node.id}:opt`, "aria-label": "Add option",
		});
		const addFrom = (text) => {
			const fresh = text.split(/[,\n]/).map((t) => t.trim()).filter((t) => t && !node.options.includes(t));
			if (!fresh.length) return false;
			node.options.push(...new Set(fresh));
			commit(true);
			return true;
		};
		input.addEventListener("keydown", (e) => {
			if ((e.key === "Enter" || e.key === ",") && input.value.trim()) {
				e.preventDefault();
				addFrom(input.value);
			} else if (e.key === "Backspace" && !input.value && node.options.length) {
				node.options.pop();
				commit(true);
			}
		});
		input.addEventListener("blur", () => { if (input.value.trim()) addFrom(input.value); });
		input.addEventListener("paste", (e) => {
			const text = e.clipboardData.getData("text");
			if (/[,\n]/.test(text)) { e.preventDefault(); addFrom(text); }
		});
		return h("div", { class: "options-editor", onclick: (e) => { if (e.target === e.currentTarget) input.focus(); } },
			node.options.map((opt, i) => h("span", { class: "chip" }, opt,
				h("button", { type: "button", class: "x", title: `Remove "${opt}"`, "aria-label": `Remove ${opt}`, onclick: () => { node.options.splice(i, 1); commit(true); } }, icon("x")))),
			input,
		);
	}

	function numberInput(node, key, label) {
		const input = h("input", {
			type: "number", value: node[key] ?? "", placeholder: "none", "data-focus": `${node.id}:${key}`,
			"aria-label": `${label} for ${node.name || "field"}`,
		});
		input.addEventListener("input", () => {
			node[key] = input.value === "" ? null : Number(input.value);
			commit(false);
		});
		return h("label", { class: "inline-field" }, h("span", {}, label), input);
	}

	function kindSpecific(node) {
		if (node.kind === "choice" || node.kind === "tags") return optionsEditor(node);
		if (node.kind === "number" || node.kind === "integer") {
			return h("div", { class: "row" }, numberInput(node, "minimum", "Min"), numberInput(node, "maximum", "Max"));
		}
		if (node.kind === "list") {
			const current = node.item ? node.item.kind || "any" : "any";
			const select = h("select", { "aria-label": "List item type" }, ITEM_KINDS.map(([v, label]) => h("option", { value: v, selected: v === current }, label)));
			select.addEventListener("change", () => {
				node.item = select.value === "any" ? null : newNode(select.value);
				commit(true);
			});
			return h("div", { class: "stack" },
				h("label", { class: "inline-field" }, h("span", {}, "Each item is"), select),
				node.item && node.item.kind === "object" ? fieldList(node.item) : null,
			);
		}
		if (node.kind === "object") return fieldList(node);
		return null;
	}

	function fieldRow(node, parent, index) {
		const siblings = parent.fields;
		const name = h("input", {
			type: "text", class: "code field-name", value: node.name, placeholder: "field_name", spellcheck: "false",
			"data-focus": `${node.id}:name`, "aria-label": "Field name",
		});
		name.addEventListener("input", () => { node.name = name.value.trim(); commit(false); });
		const kind = h("select", { class: "field-kind", "aria-label": "Field type" }, KINDS.map(([v, label]) => h("option", { value: v, selected: v === node.kind }, label)));
		kind.addEventListener("change", () => {
			siblings[index] = changeKind(node, kind.value);
			commit(true);
		});
		const desc = h("input", {
			type: "text", class: "field-desc", value: node.description, placeholder: "Describe it for the model (optional)",
			"data-focus": `${node.id}:desc`, "aria-label": "Description",
		});
		desc.addEventListener("input", () => { node.description = desc.value; commit(false); });
		const grip = h("span", { class: "grip", title: "Drag to reorder" }, icon("grip"));
		const kindIcon = (KINDS.find(([v]) => v === node.kind) || [])[2] || "braces";

		const row = h("div", { class: `field-row kind-${node.kind}`, dataset: { id: node.id } },
			h("div", { class: "field-main" },
				grip,
				h("span", { class: "kind-badge", title: (KINDS.find(([v]) => v === node.kind) || [])[1] }, icon(kindIcon)),
				name,
				kind,
				menu(button(null, { icon: "more", kind: "ghost", size: "sm", title: "Field actions" }), [
					{ label: "Move up", icon: "arrow-up", disabled: index === 0, onclick: () => move(siblings, index, index - 1) },
					{ label: "Move down", icon: "arrow-down", disabled: index === siblings.length - 1, onclick: () => move(siblings, index, index + 1) },
					"-",
					{ label: "Delete field", icon: "trash", danger: true, onclick: () => { siblings.splice(index, 1); commit(true); } },
				]),
			),
			h("div", { class: "field-body" },
				kindSpecific(node),
				h("div", { class: "field-foot" },
					desc,
					toggleSwitch("Required", node.required, (v) => { node.required = v; commit(false); }, { title: "The model must always include this field" }),
					toggleSwitch("Can be empty", node.nullable, (v) => { node.nullable = v; commit(false); }, { title: "Allow null when the input doesn't contain it" }),
				),
				h("div", { class: "field-msg", hidden: true }),
			),
		);

		grip.addEventListener("mousedown", () => { row.draggable = true; });
		row.addEventListener("dragstart", (e) => {
			if (!row.draggable) return;
			e.stopPropagation();
			drag = { list: siblings, from: index };
			e.dataTransfer.effectAllowed = "move";
			row.classList.add("dragging");
		});
		row.addEventListener("dragend", () => {
			row.draggable = false;
			row.classList.remove("dragging");
			for (const r of $$(".drop-before, .drop-after", el)) r.classList.remove("drop-before", "drop-after");
			drag = null;
		});
		row.addEventListener("dragover", (e) => {
			if (!drag || drag.list !== siblings) return;
			e.preventDefault();
			e.stopPropagation();
			const after = e.offsetY > row.offsetHeight / 2;
			row.classList.toggle("drop-before", !after);
			row.classList.toggle("drop-after", after);
		});
		row.addEventListener("dragleave", () => row.classList.remove("drop-before", "drop-after"));
		row.addEventListener("drop", (e) => {
			if (!drag || drag.list !== siblings) return;
			e.preventDefault();
			e.stopPropagation();
			const after = row.classList.contains("drop-after");
			let to = index + (after ? 1 : 0);
			if (drag.from < to) to--;
			if (to !== drag.from) move(siblings, drag.from, to);
		});
		return row;
	}

	function fieldList(parent) {
		const add = () => {
			const node = newNode("text");
			parent.fields.push(node);
			commit(true);
			const input = $(`[data-focus="${node.id}:name"]`, el);
			if (input) input.focus();
		};
		return h("div", { class: parent === root ? "field-list" : "field-list nested" },
			parent.fields.map((f, i) => fieldRow(f, parent, i)),
			parent === root && !parent.fields.length
				? h("div", { class: "fields-empty" }, "No fields yet. The model's answer is a JSON object with the fields you add here.")
				: null,
			button(parent === root ? "Add field" : "Add nested field", { icon: "plus", kind: parent === root ? "secondary" : "ghost", size: "sm", onclick: add, cls: "add-field" }),
		);
	}

	function render() {
		mount(el,
			h("div", { class: "callout bad schema-banner", hidden: !issues.errors.size }, icon("alert-circle"),
				h("div", {}, "Fix the highlighted fields. Changes are saved once every field is valid.")),
			fieldList(root),
		);
		showIssues();
	}

	render();
	return el;
}
