import { h, icon } from "./dom.js";
import { labelText, plural } from "./format.js";
import { S, duplicateCounts } from "./state.js";

const TARGET = 100;

const sameValue = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const isFiltered = (field, value) => Boolean(S.query.label && S.query.label.field === field && sameValue(S.query.label.value, value));

function bar({ field, kind, value, count, max, tone, title }, onFilter) {
	const active = isFiltered(field, value);
	return h("button", {
		type: "button", class: `health-bar ${tone || ""}`, "aria-pressed": String(active),
		title: title || (active ? "Show all examples" : `Show only ${labelText(value)}`),
		onclick: () => onFilter(active ? null : { field, value, kind }),
	},
		h("span", { class: "hb-label ellipsis" }, labelText(value)),
		h("span", { class: "hb-track" }, h("span", { style: { width: `${(100 * count) / Math.max(1, max)}%` } })),
		h("span", { class: "hb-count num" }, count === 0 ? "none" : count.toLocaleString()),
	);
}

function labelBars(f, onFilter) {
	const max = Math.max(1, ...f.values.map((v) => v.count));
	return h("div", { class: "health-field" },
		h("div", { class: "health-field-name" }, h("code", {}, f.field), f.kind === "tags" ? h("span", { class: "faint small" }, "examples per tag") : null),
		h("div", { class: "health-bars" }, f.values.map((v) => bar({
			field: f.field, kind: f.kind, value: v.value, count: v.count, max,
			tone: v.count === 0 ? "bad" : v.count < 3 ? "warn" : "",
			title: v.count === 0 ? "No examples: the model can't learn a value it never sees" : null,
		}, onFilter))),
	);
}

function refBars(r, onFilter) {
	const max = Math.max(1, ...r.top.map((v) => v.count));
	const coverage = r.size ? r.distinct / r.size : 0;
	return h("div", { class: "health-field" },
		h("div", { class: "health-field-name" }, h("code", {}, r.field),
			h("span", { class: "faint small" }, `${r.distinct.toLocaleString()} of ${r.size.toLocaleString()} values in ${r.name} used (${(coverage * 100).toFixed(coverage < 0.01 ? 2 : 1)}%)`)),
		r.missing ? h("span", { class: "pill bad", title: "Values that aren't in the data set; filter To fix to find them" }, icon("alert"), `${plural(r.missing, "value")} not in ${r.name}`) : null,
		h("div", { class: "health-bars" }, r.top.map((v) => bar({ field: r.field, kind: "ref", value: v.value, count: v.count, max }, onFilter))),
		r.distinct > r.top.length ? h("p", { class: "faint small" }, `The ${r.top.length} most used values. Values with one or two examples are the hardest to learn.`) : null,
	);
}

// Dataset overview: progress toward a healthy size, problems, duplicates and per-value counts.
export function healthPanel({ onFilter, onShow }) {
	const n = S.counts.all;
	const bad = S.counts.problems;
	const { conflicts, plain } = duplicateCounts();
	const stats = S.stats || { fields: [], refs: [] };
	const meter = Math.min(1, n / TARGET);
	const advice = n < 20 ? "Aim for at least 20, ideally 100+." : n < TARGET ? "Good start. 100+ covering every label works best." : "A healthy amount.";
	const checking = S.checking.running;
	const allChecked = S.checking.done >= S.checking.total;
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
				checking ? h("span", { class: "pill info dot live" }, `Checking ${S.checking.done.toLocaleString()} of ${S.checking.total.toLocaleString()}`) : null,
				S.checking.error ? h("span", { class: "pill bad", title: S.checking.error }, icon("alert-circle"), "Checking stopped: ", S.checking.error) : null,
				bad ? h("button", { type: "button", class: "chip bad-chip", onclick: () => onShow("problems") }, icon("alert-circle"), plural(bad, "example"), " to fix") : null,
				conflicts || plain ? h("button", { type: "button", class: "chip warn-chip", onclick: () => onShow("duplicates"), title: "The same input more than once" + (conflicts ? ", sometimes with different answers" : "") },
					icon("copy"), plural(conflicts + plain, "repeated input"), conflicts ? ` (${conflicts} conflicting)` : "") : null,
				!checking && allChecked && !bad && !conflicts && !plain && n ? h("span", { class: "pill good", title: "Every example matches the schema" }, icon("check"), "All valid") : null,
			),
		),
		stats.fields.length || stats.refs.length ? h("div", { class: "health-fields" },
			stats.fields.map((f) => labelBars(f, onFilter)),
			stats.refs.map((r) => refBars(r, onFilter)),
		) : null,
	);
}
