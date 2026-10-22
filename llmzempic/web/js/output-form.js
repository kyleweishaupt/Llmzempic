import { button, h, icon, mount, preserveFocus, seg } from "./dom.js";
import { blankValue, labelFields } from "./schema-model.js";

const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);

function autoGrow(textarea) {
	const fit = () => {
		textarea.style.height = "auto";
		textarea.style.height = `${textarea.scrollHeight + 2}px`;
	};
	textarea.addEventListener("input", fit);
	requestAnimationFrame(fit);
	return textarea;
}

function choiceControl(node, current, set, label) {
	const options = node.kind === "boolean" ? [[true, "Yes"], [false, "No"]] : node.options.map((o) => [o, o]);
	if (node.nullable) options.push([null, "Empty"]);
	const short = options.length <= 4 && options.reduce((n, [, text]) => n + text.length, 0) <= 40;
	if (short) return seg(options, current, set, { accent: true, label });
	const select = h("select", { "aria-label": label },
		h("option", { value: "", disabled: true, selected: current === undefined }, "Choose…"),
		options.map(([v, text], i) => h("option", { value: String(i), selected: v === current }, text)));
	select.addEventListener("change", () => set(options[Number(select.value)][0]));
	return select;
}

function tagControl(node, current, set, label) {
	const chosen = new Set(Array.isArray(current) ? current : []);
	return h("div", { class: "tag-picks", role: "group", "aria-label": label }, node.options.map((opt) => h("button", {
		type: "button", class: "chip", "aria-pressed": String(chosen.has(opt)),
		onclick: () => {
			if (chosen.has(opt)) chosen.delete(opt);
			else chosen.add(opt);
			set(node.options.filter((o) => chosen.has(o)));
		},
	}, chosen.has(opt) ? icon("check") : null, opt)));
}

// Edits `value` (an example's output object) in place, calling onChange(value) after each edit.
export function outputForm(root, value, onChange, { labelsOnly = false, keyPrefix = "of" } = {}) {
	const el = h("div", { class: labelsOnly ? "output-form labels-only" : "output-form" });

	function control(node, get, set, path, label) {
		const key = `${keyPrefix}:${path}`;
		const current = get();
		if (node.kind === "boolean" || node.kind === "choice") {
			return choiceControl(node, current, (v) => { set(v); rerender(); }, label);
		}
		if (node.kind === "tags") return tagControl(node, current, (v) => { set(v); rerender(); }, label);
		if (node.kind === "text") {
			const input = h("textarea", { rows: 1, value: typeof current === "string" ? current : current == null ? "" : JSON.stringify(current), "data-focus": key, "aria-label": label, placeholder: node.description || "" });
			input.addEventListener("input", () => set(input.value === "" && node.nullable ? null : input.value));
			return autoGrow(input);
		}
		if (node.kind === "number" || node.kind === "integer") {
			const input = h("input", { type: "number", step: node.kind === "integer" ? 1 : "any", value: typeof current === "number" ? current : "", "data-focus": key, "aria-label": label, min: node.minimum ?? null, max: node.maximum ?? null });
			input.addEventListener("input", () => set(input.value === "" ? (node.nullable ? null : undefined) : Number(input.value)));
			return input;
		}
		if (node.kind === "object") {
			if (!isObj(current)) {
				return button(current === null ? "Fill in" : "Add", { icon: "plus", kind: "ghost", size: "sm", onclick: () => { set(blankValue(node)); rerender(); } });
			}
			return fields(node, current, path);
		}
		if (node.kind === "list") {
			const list = Array.isArray(current) ? current : [];
			if (!node.item) return jsonControl(current, set, key, label);
			return h("div", { class: "of-list" },
				list.map((item, i) => h("div", { class: node.item.kind === "object" ? "of-item block" : "of-item" },
					control(node.item, () => list[i], (v) => { list[i] = v; set(list); }, `${path}.${i}`, `${label} item ${i + 1}`),
					button(null, { icon: "x", kind: "ghost", size: "sm", title: "Remove item", onclick: () => { list.splice(i, 1); set(list); rerender(); } }),
				)),
				button("Add item", { icon: "plus", kind: "ghost", size: "sm", onclick: () => {
					list.push(node.item.kind === "object" ? blankValue(node.item) : node.item.kind === "text" ? "" : node.item.kind === "boolean" ? false : 0);
					set(list);
					rerender();
				} }),
			);
		}
		return jsonControl(current, set, key, label);
	}

	function jsonControl(current, set, key, label) {
		const input = h("textarea", { class: "code", rows: 1, value: current === undefined ? "" : JSON.stringify(current), "data-focus": key, "aria-label": label });
		input.addEventListener("input", () => {
			try {
				set(input.value.trim() ? JSON.parse(input.value) : undefined);
				input.classList.remove("invalid");
			} catch {
				input.classList.add("invalid");
			}
		});
		return autoGrow(input);
	}

	function fields(node, obj, path) {
		const shown = labelsOnly && node === root ? labelFields(node) : node.fields;
		return h("div", { class: node === root ? "of-fields" : "of-fields nested" }, shown.map((f) => {
			const set = (v) => {
				if (v === undefined) delete obj[f.name];
				else obj[f.name] = v;
				onChange(value);
			};
			const wide = ["text", "object", "list", "tags"].includes(f.kind);
			return h("div", { class: wide ? "of-field wide" : "of-field" },
				h("span", { class: "of-label", title: f.description || null }, f.name, f.required ? null : h("span", { class: "faint" }, " optional")),
				h("div", { class: "of-control" }, control(f, () => obj[f.name], set, `${path}.${f.name}`, f.name)),
			);
		}));
	}

	function rerender() {
		preserveFocus(() => mount(el, fields(root, value, "")));
	}

	mount(el, fields(root, value, ""));
	return el;
}
