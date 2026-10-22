import { api } from "./api.js";
import { h, icon, mount } from "./dom.js";
import { S } from "./state.js";

const datasetName = (id) => ((S.datasets || []).find((d) => d.id === id) || {}).name || id;

// Search-as-you-type picker for values from a data set. With list: true the chosen values are
// chips in order (first is first), reorderable by drag or Alt+arrow keys.
export function refPicker({ dataset, value, list = false, max = null, onChange, label }) {
	const el = h("div", { class: "ref-picker" });
	let values = list ? (Array.isArray(value) ? [...value] : []) : (typeof value === "string" && value ? [value] : []);
	let results = [];
	let failed = null;
	let active = -1;
	let timer = null;
	let dragFrom = null;

	const input = h("input", {
		type: "search", class: "ref-input", autocomplete: "off", spellcheck: "false",
		placeholder: `Search ${datasetName(dataset)} by value or words`, "aria-label": label || `Search ${datasetName(dataset)}`,
		role: "combobox", "aria-expanded": "false", "aria-autocomplete": "list",
	});
	const menu = h("div", { class: "ref-results", role: "listbox", hidden: true });

	function commit() {
		onChange(list ? [...values] : values[0]);
		draw();
	}

	function choose(v) {
		if (list) {
			if (!values.includes(v)) values.push(v);
		} else {
			values = [v];
		}
		input.value = "";
		close();
		commit();
		input.focus();
	}

	function close() {
		menu.hidden = true;
		input.setAttribute("aria-expanded", "false");
		active = -1;
	}

	function showResults() {
		mount(menu, results.length
			? results.map((r, i) => h("button", {
				type: "button", role: "option", class: "ref-option", "aria-selected": String(i === active),
				onmousedown: (e) => { e.preventDefault(); choose(r.value); },
			}, h("code", {}, r.value), r.description ? h("span", { class: "muted" }, r.description) : null))
			: h("div", { class: "ref-empty faint" }, failed ? `Couldn't search: ${failed}` : "No matching values."));
		menu.hidden = false;
		input.setAttribute("aria-expanded", "true");
	}

	async function search() {
		const q = input.value.trim();
		try {
			results = (await api(`/datasets/${encodeURIComponent(dataset)}/search?q=${encodeURIComponent(q)}&limit=12`)).rows;
			failed = null;
		} catch (e) {
			results = [];
			failed = e.message;
		}
		active = results.length ? 0 : -1;
		if (document.activeElement === input) showResults();
	}

	input.addEventListener("input", () => {
		clearTimeout(timer);
		timer = setTimeout(search, 150);
	});
	input.addEventListener("focus", () => { if (!results.length) search(); else showResults(); });
	input.addEventListener("blur", () => setTimeout(close, 120));
	input.addEventListener("keydown", (e) => {
		if (e.key === "ArrowDown" || e.key === "ArrowUp") {
			e.preventDefault();
			if (menu.hidden) { showResults(); return; }
			active = (active + (e.key === "ArrowDown" ? 1 : -1) + results.length) % Math.max(1, results.length);
			showResults();
		} else if (e.key === "Enter" && !menu.hidden && results[active]) {
			e.preventDefault();
			choose(results[active].value);
		} else if (e.key === "Escape") {
			close();
		} else if (e.key === "Backspace" && !input.value && values.length && list) {
			values.pop();
			commit();
		}
	});

	function move(from, to) {
		if (to < 0 || to >= values.length) return;
		const [v] = values.splice(from, 1);
		values.splice(to, 0, v);
		commit();
		el.querySelectorAll(".ref-chip")[to].focus();
	}

	// Keep keyboard focus nearby: the next chip, or the search box once none are left.
	function remove(i) {
		values.splice(i, 1);
		commit();
		const chips = el.querySelectorAll(".ref-chip");
		(chips[Math.min(i, chips.length - 1)] || input).focus();
	}

	function chip(v, i) {
		const el = h("span", {
			class: "chip ref-chip", tabindex: 0, draggable: list ? "true" : null, title: list ? "Drag to reorder (Alt+arrows on the keyboard)" : null,
			onkeydown: (e) => {
				if (list && e.altKey && (e.key === "ArrowLeft" || e.key === "ArrowRight")) { e.preventDefault(); move(i, i + (e.key === "ArrowRight" ? 1 : -1)); }
				if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); remove(i); }
			},
		},
			list ? h("span", { class: "ref-order num" }, String(i + 1)) : null,
			h("code", {}, v),
			h("button", { type: "button", class: "x", "aria-label": `Remove ${v}`, onclick: () => remove(i) }, icon("x")),
		);
		if (list) {
			el.addEventListener("dragstart", (e) => { dragFrom = i; e.dataTransfer.effectAllowed = "move"; });
			el.addEventListener("dragover", (e) => { if (dragFrom !== null) e.preventDefault(); });
			el.addEventListener("drop", (e) => { e.preventDefault(); if (dragFrom !== null) move(dragFrom, i); dragFrom = null; });
		}
		return el;
	}

	function draw() {
		const focused = document.activeElement === input;
		const full = list && max !== null && values.length >= max;
		input.disabled = full;
		input.placeholder = full ? `Holds at most ${max}` : `Search ${datasetName(dataset)} by value or words`;
		mount(el,
			values.length ? h("div", { class: "ref-chips" }, values.map(chip)) : null,
			h("div", { class: "ref-search" }, icon("search"), input, menu),
		);
		if (focused && !full) input.focus();
	}

	draw();
	return el;
}
