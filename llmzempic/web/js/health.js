import { h, icon } from "./dom.js";
import { labelText, plural } from "./format.js";
import { S, duplicateCounts, problemCount } from "./state.js";

const TARGET = 100;

const sameValue = (a, b) => JSON.stringify(a) === JSON.stringify(b);

export const isLabelFilter = (field, value) => Boolean(S.labelFilter && S.labelFilter.field === field && sameValue(S.labelFilter.value, value));

function labelBars(f, onFilter) {
	const max = Math.max(1, ...f.values.map((v) => v.count));
	return h("div", { class: "health-field" },
		h("div", { class: "health-field-name" }, h("code", {}, f.field), f.kind === "tags" ? h("span", { class: "faint small" }, "examples per tag") : null),
		h("div", { class: "health-bars" }, f.values.map((v) => {
			const tone = v.count === 0 ? "bad" : v.count < 3 ? "warn" : "";
			const active = isLabelFilter(f.field, v.value);
			return h("button", {
				type: "button", class: `health-bar ${tone}`, "aria-pressed": String(active),
				title: v.count === 0 ? "No examples yet: the model can't learn a value it never sees" : active ? "Show all examples" : `Show only ${labelText(v.value)}`,
				onclick: () => onFilter(active ? null : { field: f.field, value: v.value, kind: f.kind }),
			},
				h("span", { class: "hb-label ellipsis" }, labelText(v.value)),
				h("span", { class: "hb-track" }, h("span", { style: { width: `${(100 * v.count) / max}%` } })),
				h("span", { class: "hb-count num" }, v.count === 0 ? "none" : String(v.count)),
			);
		})),
	);
}

// Dataset overview: progress toward a healthy size, problems, duplicates and per-label counts.
export function healthPanel({ onFilter, onShow }) {
	const n = S.examples.length;
	const bad = problemCount();
	const { conflicts, plain } = duplicateCounts();
	const fields = (S.stats && S.stats.fields) || [];
	const meter = Math.min(1, n / TARGET);
	const advice = n < 20 ? "Aim for at least 20, ideally 100+." : n < TARGET ? "Good start. 100+ covering every label works best." : "A healthy amount.";
	return h("section", { class: "card health" },
		h("div", { class: "health-summary" },
			h("div", { class: "health-total" },
				h("span", { class: "big num" }, n.toLocaleString()),
				h("span", { class: "muted" }, n === 1 ? "example" : "examples"),
			),
			h("div", { class: "health-meter" },
				h("div", { class: "progress" }, h("span", { style: { width: `${meter * 100}%` } })),
				h("span", { class: "faint small" }, advice),
			),
			h("div", { class: "row" },
				bad ? h("button", { type: "button", class: "chip bad-chip", onclick: () => onShow("problems") }, icon("alert-circle"), plural(bad, "example"), " to fix") : null,
				conflicts || plain ? h("button", { type: "button", class: "chip warn-chip", onclick: () => onShow("duplicates"), title: "The same input more than once" + (conflicts ? ", sometimes with different answers" : "") },
					icon("copy"), plural(conflicts + plain, "repeated input"), conflicts ? ` (${conflicts} conflicting)` : "") : null,
				!bad && !conflicts && !plain && n ? h("span", { class: "pill good", title: "Every example matches the schema" }, icon("check"), "All valid") : null,
			),
		),
		fields.length ? h("div", { class: "health-fields" }, fields.map((f) => labelBars(f, onFilter))) : null,
	);
}

export function matchesLabel(example, filter) {
	const out = example.output;
	if (!out || typeof out !== "object" || !(filter.field in out)) return false;
	const v = out[filter.field];
	return filter.kind === "tags" ? Array.isArray(v) && v.some((x) => sameValue(x, filter.value)) : sameValue(v, filter.value);
}
