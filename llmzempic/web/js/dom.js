import { ICONS } from "./icons.js";

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

// Native append/replaceChildren turn null into the text "null", so every child list goes through this.
function nodes(children) {
	const out = [];
	for (const c of children.flat(Infinity)) {
		if (c === null || c === undefined || c === false || c === true) continue;
		out.push(c instanceof Node ? c : document.createTextNode(String(c)));
	}
	return out;
}

export function mount(el, ...children) {
	el.replaceChildren(...nodes(children));
	return el;
}

export function add(el, ...children) {
	el.append(...nodes(children));
	return el;
}

const PROPS = new Set(["value", "checked", "disabled", "selected", "open", "indeterminate", "hidden", "multiple"]);

export function h(tag, attrs, ...children) {
	const el = document.createElement(tag);
	for (const [k, v] of Object.entries(attrs || {})) {
		if (v === null || v === undefined || v === false) continue;
		if (k === "class") el.className = v;
		else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
		else if (k === "dataset") Object.assign(el.dataset, v);
		else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
		else if (PROPS.has(k)) el[k] = v;
		else el.setAttribute(k, v === true ? "" : v);
	}
	return add(el, children);
}

const SVG_NS = "http://www.w3.org/2000/svg";

export function s(tag, attrs, ...children) {
	const el = document.createElementNS(SVG_NS, tag);
	for (const [k, v] of Object.entries(attrs || {})) if (v !== null && v !== undefined) el.setAttribute(k, v);
	el.append(...nodes(children));
	return el;
}

export function icon(name, cls) {
	const el = document.createElementNS(SVG_NS, "svg");
	el.setAttribute("viewBox", "0 0 24 24");
	el.setAttribute("aria-hidden", "true");
	el.setAttribute("class", cls ? `icon ${cls}` : "icon");
	el.innerHTML = ICONS[name] || "";
	return el;
}

// Re-render a container without losing the caret: elements tagged data-focus keep focus by key.
export function preserveFocus(render) {
	const active = document.activeElement;
	const key = active && active.dataset && active.dataset.focus;
	const caret = key && "selectionStart" in active ? [active.selectionStart, active.selectionEnd] : null;
	render();
	if (!key) return;
	const next = document.querySelector(`[data-focus="${CSS.escape(key)}"]`);
	if (!next) return;
	next.focus();
	if (caret && "setSelectionRange" in next) {
		try { next.setSelectionRange(...caret); } catch { /* number inputs have no caret */ }
	}
}

export function button(label, { icon: name, kind = "secondary", size, title, onclick, disabled, type = "button", cls } = {}) {
	const classes = ["btn", `btn-${kind}`];
	if (size) classes.push(`btn-${size}`);
	if (!label) classes.push("btn-icon");
	if (cls) classes.push(cls);
	return h("button", {
		type, class: classes.join(" "), title, disabled, onclick,
		"aria-label": !label ? title : null,
	}, name ? icon(name) : null, label);
}

export function pill(text, tone, { dot = false, live = false, iconName } = {}) {
	const cls = ["pill", tone, dot ? "dot" : "", live ? "live" : ""].filter(Boolean).join(" ");
	return h("span", { class: cls }, iconName ? icon(iconName) : null, text);
}

export function seg(options, current, onPick, { accent = false, label, disabled = false } = {}) {
	return h("div", { class: accent ? "seg seg-accent" : "seg", role: "group", "aria-label": label },
		options.map(([value, text, iconName]) => h("button", {
			type: "button",
			"aria-pressed": String(value === current),
			disabled,
			onclick: () => onPick(value),
		}, iconName ? icon(iconName) : null, text)));
}

export function toggleSwitch(label, checked, onChange, { disabled = false, title } = {}) {
	const input = h("input", { type: "checkbox", checked, disabled, onchange: () => onChange(input.checked) });
	return h("label", { class: "switch", title }, input, h("span", { class: "track" }), label);
}

export function field(label, control, { help, hint, error } = {}) {
	return h("label", { class: "field" },
		h("span", { class: "field-label" }, label, hint ? h("span", { class: "faint" }, hint) : null),
		control,
		help ? h("span", { class: "field-help" }, help) : null,
		error ? h("span", { class: "field-error" }, error) : null,
	);
}

export function callout(tone, iconName, ...children) {
	return h("div", { class: `callout ${tone}` }, icon(iconName), h("div", { class: "grow" }, children));
}

export function emptyState(iconName, title, text, ...actions) {
	return h("div", { class: "empty" },
		h("div", { class: "empty-icon" }, icon(iconName)),
		h("h3", {}, title),
		text ? h("p", {}, text) : null,
		actions.length ? h("div", { class: "row" }, actions) : null,
	);
}

export function spinnerLine(text) {
	return h("div", { class: "loading" }, h("span", { class: "spinner" }), text);
}

export function menu(trigger, items) {
	const details = h("details", { class: "menu" });
	const close = () => details.removeAttribute("open");
	const list = h("div", { class: "menu-list", role: "menu" }, items.map((it) => {
		if (it === "-") return h("hr");
		if (!it) return null;
		return h("button", {
			type: "button", role: "menuitem", class: it.danger ? "danger" : "", disabled: it.disabled,
			onclick: () => { close(); it.onclick(); },
		}, it.icon ? icon(it.icon) : null, it.label);
	}));
	const summary = h("summary", { class: trigger.className, title: trigger.title, "aria-label": trigger.getAttribute("aria-label") }, [...trigger.childNodes]);
	add(details, summary, list);
	return details;
}

// Close open menus when clicking elsewhere or pressing Escape.
document.addEventListener("click", (e) => {
	for (const d of $$("details.menu[open]")) if (!d.contains(e.target)) d.removeAttribute("open");
});
document.addEventListener("keydown", (e) => {
	if (e.key === "Escape") for (const d of $$("details.menu[open]")) d.removeAttribute("open");
});
