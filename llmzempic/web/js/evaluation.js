import { runImageUrl } from "./api.js";
import { h, icon, mount, pill, toggleSwitch } from "./dom.js";
import { toastError } from "./feedback.js";
import { clip, labelText, pct } from "./format.js";
import { jsonNodes } from "./json-view.js";
import { loadPage } from "./project.js";
import { S, app } from "./state.js";

function tone(rate) {
	if (rate === null || rate === undefined) return "";
	return rate >= 0.9 ? "good" : rate >= 0.6 ? "warn" : "bad";
}

// Calibration error: how far confidence is from accuracy (lower is better; 0.05 or less is well calibrated).
const eceTone = (v) => (v <= 0.05 ? "good" : v <= 0.1 ? "warn" : "bad");
const fixed = (v, digits = 3) => (v === null || v === undefined ? "–" : Number(v).toFixed(digits));

function decisionTiles(m) {
	const d = m.decisions;
	const cal = d.after_calibration || d;
	return h("div", { class: "stack" },
		h("div", { class: "metrics" },
			h("div", { class: `metric hero ${tone(d.accuracy)}` },
				h("span", { class: "metric-value num" }, pct(d.accuracy)),
				h("span", { class: "metric-label" }, "Accuracy"),
				h("span", { class: "faint small" }, "Top answers that are right, over all questions"),
			),
			h("div", { class: `metric ${eceTone(cal.ece)}`, title: "Expected calibration error after temperature scaling: the gap between confidence and accuracy" },
				h("span", { class: "metric-value num" }, fixed(cal.ece)),
				h("span", { class: "metric-label" }, "Calibration error"),
			),
			d.baseline ? h("div", { class: "metric", title: "The base model reading the same prompts without fine-tuning" },
				h("span", { class: "metric-value num" }, pct(d.baseline.accuracy)),
				h("span", { class: "metric-label" }, "Base model"),
			) : null,
			d.latency_ms ? h("div", { class: "metric", title: "Time to read every question for one input, on the training machine" },
				h("span", { class: "metric-value num" }, `${Math.round(d.latency_ms.p50)} ms`),
				h("span", { class: "metric-label" }, "Per input (p50)"),
			) : null,
		),
		h("p", { class: "faint small" }, `Scored ${d.count} answers on ${m.count} ${m.eval_on || "examples"}; all questions right on ${pct(m.exact_match)}. `,
			d.calibrated
				? `Probabilities are calibrated with temperature ${d.temperature} (calibration error ${fixed(d.ece)} before). `
				: "Probabilities aren't rescaled: 50 or more held-out answers are needed to fit a calibration temperature. ",
			`About ${d.input_tokens} input tokens per input; nothing is generated.`),
	);
}

function decisionBreakdown(d) {
	if (!d || !d.questions || !d.questions.length) return null;
	const base = Object.fromEntries(((d.baseline && d.baseline.questions) || []).map((q) => [q.name, q]));
	return h("div", { class: "breakdown stack" },
		h("h4", {}, "Per question"),
		h("div", { class: "table-wrap" }, h("table", { class: "decision-table" },
			h("thead", {}, h("tr", {},
				h("th", {}, "Question"),
				h("th", { class: "num" }, "Answers"),
				h("th", {}, "Accuracy"),
				h("th", { class: "num", title: "Accuracy of the base model without fine-tuning" }, "Base"),
				h("th", { class: "num", title: "Expected calibration error after temperature scaling (lower is better)" }, "Calibration"),
				h("th", { class: "num", title: "Brier score (lower is better)" }, "Brier"),
			)),
			h("tbody", {}, d.questions.map((q) => h("tr", {},
				h("td", {}, h("code", {}, q.name), h("span", { class: "faint" }, ` ${q.type}`), q.mae !== undefined ? h("span", { class: "faint small", title: "Mean distance between the expected score and the right level" }, ` ±${fixed(q.mae, 2)} levels`) : null),
				h("td", { class: "num" }, String(q.count)),
				rateBar(Math.round(q.accuracy * q.count), q.count),
				h("td", { class: "num" }, base[q.name] ? pct(base[q.name].accuracy) : "–"),
				h("td", { class: `num ${eceTone(q.ece)}-text` }, fixed(q.ece)),
				h("td", { class: "num" }, fixed(q.brier)),
			))),
		)),
	);
}

export function metricTiles(m) {
	if (m.decisions) return decisionTiles(m);
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

async function showExamples(field, value) {
	S.query = { filter: "all", search: "", label: { field, value, kind: "ref" }, offset: 0 };
	S.selection.clear();
	S.selectAll = false;
	try {
		await loadPage({ stats: false });
	} catch (e) {
		toastError(e);
		return;
	}
	app.showTab("examples");
}

const REF_TILES = [
	["exact", "Exact", "Every value right, in order"],
	["first", "First value", "The first value right"],
	["f1", "F1", "Balance of precision and recall"],
	["precision", "Precision", "Values the model gave that were expected"],
	["recall", "Recall", "Expected values the model gave"],
];

function refBreakdown(refs) {
	return (refs || []).map((r) => h("div", { class: "breakdown stack" },
		h("h4", {}, h("code", {}, r.field), h("span", { class: "faint" }, ` values from ${r.name}`)),
		h("div", { class: "metrics ref-metrics" },
			REF_TILES.map(([key, label, hint]) => h("div", { class: `metric ${tone(r[key])}`, title: hint },
				h("span", { class: "metric-value num" }, pct(r[key])), h("span", { class: "metric-label" }, label))),
			h("div", { class: `metric ${r.invalid_rate ? "bad" : "good"}`, title: "Answers with values that aren't in the data set" },
				h("span", { class: "metric-value num" }, pct(r.invalid_rate)), h("span", { class: "metric-label" }, "Not in the set")),
		),
		r.values.length ? h("details", { class: "ref-values" },
			h("summary", {}, icon("list"), `Per value (${r.values.length} most common)`),
			h("div", { class: "table-wrap" }, h("table", {},
				h("thead", {}, h("tr", {},
					h("th", {}, "Value"),
					h("th", { class: "num", title: "Scored examples whose expected answer has this value" }, "Examples"),
					h("th", { title: "How many of those the model answered with this value" }, "Recognized"),
					h("th", { title: "When the model gave this value, how often it was right" }, "Precision"),
					h("th", {}, h("span", { class: "sr-only" }, "Examples")),
				)),
				h("tbody", {}, r.values.map((v) => h("tr", {},
					h("td", {}, h("code", {}, v.value)),
					h("td", { class: "num" }, String(v.expected)),
					rateBar(v.correct, v.expected),
					rateBar(v.correct, v.predicted),
					h("td", {}, h("button", { type: "button", class: "link-btn small nowrap", onclick: () => showExamples(r.field, v.value) }, "Show examples")),
				))),
			))) : null,
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
		decisionBreakdown(ev.decisions),
		fields.length > 1 && !ev.decisions ? h("div", { class: "field-acc" }, h("span", { class: "muted small" }, "Per field"),
			fields.map(([k, v]) => h("span", { class: `chip ${tone(v)}-chip-lite` }, h("code", {}, k), h("span", { class: "count" }, pct(v))))) : null,
		labelBreakdown(ev.labels),
		refBreakdown(ev.refs),
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
