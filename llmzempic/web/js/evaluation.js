import { runImageUrl } from "./api.js";
import { h, icon, mount, pill, toggleSwitch } from "./dom.js";
import { clip, labelText, pct } from "./format.js";
import { jsonNodes } from "./json-view.js";
import { S } from "./state.js";

function tone(rate) {
	if (rate === null || rate === undefined) return "";
	return rate >= 0.9 ? "good" : rate >= 0.6 ? "warn" : "bad";
}

export function metricTiles(m) {
	return h("div", { class: "stack" },
		h("div", { class: "metrics" },
			h("div", { class: `metric hero ${tone(m.exact_match)}` },
				h("span", { class: "metric-value num" }, pct(m.exact_match)),
				h("span", { class: "metric-label" }, "Exact match"),
				h("span", { class: "faint small" }, "Answers identical to the expected JSON"),
			),
			[["schema_valid", "Matches the schema"], ["json_valid", "Valid JSON"]].map(([k, label]) => h("div", { class: `metric ${tone(m[k])}` },
				h("span", { class: "metric-value num" }, pct(m[k])),
				h("span", { class: "metric-label" }, label),
			)),
		),
		h("p", { class: "faint small" }, `Scored on ${m.count} ${m.eval_on || "examples"}. Deployed with schema-constrained decoding every answer is valid JSON, so exact match is the number to watch.`),
	);
}

function rateBar(hits, total) {
	if (!total) return h("td", { class: "num faint" }, "–");
	const rate = hits / total;
	return h("td", {}, h("div", { class: "rate" },
		h("span", { class: `rate-bar ${tone(rate)}` }, h("span", { style: { width: `${rate * 100}%` } })),
		h("span", { class: "num" }, pct(rate)),
	));
}

function labelBreakdown(fields) {
	return (fields || []).map((f) => h("div", { class: "breakdown stack" },
		h("h4", {}, h("code", {}, f.field), f.kind === "tags" ? h("span", { class: "faint" }, " tags") : null),
		h("div", { class: "table-wrap" }, h("table", {},
			h("thead", {}, h("tr", {},
				h("th", {}, "Value"),
				h("th", { class: "num", title: "Scored examples whose expected answer has this value" }, "Examples"),
				h("th", { title: "How many of those the model answered with this value" }, "Recognized"),
				h("th", { title: "When the model gave this value, how often it was right" }, "Precision"),
			)),
			h("tbody", {}, f.values.map((v) => h("tr", {},
				h("td", {}, labelText(v.value)),
				h("td", { class: "num" }, String(v.expected)),
				rateBar(v.correct, v.expected),
				rateBar(v.correct, v.predicted),
			))),
		)),
		f.confusions.length ? h("p", { class: "muted small" }, h("b", {}, "Most common mix-ups: "),
			f.confusions.map((c, i) => h("span", { class: "nowrap" }, i ? "; " : "", `expected ${labelText(c.expected)}, got ${c.unknown ? "no valid value" : labelText(c.predicted)} (${c.count}×)`))) : null,
	));
}

const inlineJson = (v) => h("code", { class: "json-inline" }, jsonNodes(v));

function predictionRow(it) {
	const result = it.exact_match
		? pill("Exact", "good", { iconName: "check" })
		: it.schema_valid ? pill("Differs", "warn") : pill(it.json_valid ? "Off-schema" : "Not JSON", "bad", { iconName: "x" });
	let generated;
	try { generated = inlineJson(JSON.parse(it.generated)); } catch { generated = h("code", { class: "json-inline" }, clip(it.generated || "(no answer)", 300)); }
	return h("tr", { class: it.exact_match ? "" : "miss" },
		h("td", { class: "pred-input" },
			it.image ? h("img", { src: runImageUrl(S.pid, S.runId, it.image), alt: "", loading: "lazy" }) : null,
			it.input ? h("span", { title: it.input }, clip(it.input, 160)) : null),
		h("td", {}, inlineJson(it.expected)),
		h("td", {}, generated),
		h("td", { title: (it.errors || []).join("\n") || null }, result),
	);
}

export function evaluationView(ev) {
	const fields = Object.entries(ev.field_accuracy || {});
	const wrong = ev.items.filter((it) => !it.exact_match).length;
	const body = h("tbody");
	const fill = () => mount(body, ev.items.filter((it) => !S.mistakesOnly || !it.exact_match).slice(0, 200).map(predictionRow));
	fill();
	return h("div", { class: "stack-lg" },
		metricTiles(ev),
		fields.length > 1 ? h("div", { class: "field-acc" }, h("span", { class: "muted small" }, "Per field"),
			fields.map(([k, v]) => h("span", { class: `chip ${tone(v)}-chip-lite` }, h("code", {}, k), h("span", { class: "count" }, pct(v))))) : null,
		labelBreakdown(ev.labels),
		h("details", { class: "card predictions", open: wrong > 0 && wrong <= 20 },
			h("summary", {}, icon("list"), h("span", { class: "grow" }, `Predictions (${ev.items.length})`), wrong ? pill(`${wrong} not exact`, "warn") : pill("All exact", "good")),
			h("div", { class: "stack pred-body" },
				wrong && wrong < ev.items.length ? toggleSwitch("Only show mistakes", S.mistakesOnly, (v) => { S.mistakesOnly = v; fill(); }) : null,
				h("div", { class: "table-wrap" }, h("table", { class: "pred-table" },
					h("thead", {}, h("tr", {}, ["Input", "Expected", "Got", ""].map((t) => h("th", {}, t)))),
					body,
				)),
			),
		),
	);
}
