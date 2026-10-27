import { api } from "./api.js";
import { button, h, mount, pill, spinnerLine } from "./dom.js";
import { toast, toastError } from "./feedback.js";
import { pct, plural, when } from "./format.js";
import { loadPage } from "./project.js";
import { S, app } from "./state.js";

// Decisions logged by `llmzempic serve`: what the model answers in production, how that compares with the
// training labels, and what feedback says about its accuracy.
const fixed = (v, digits = 2) => (v === null || v === undefined ? "–" : Number(v).toFixed(digits));

function distribution(answers) {
	const total = Object.values(answers).reduce((a, b) => a + b, 0);
	const top = Object.entries(answers).slice(0, 3).map(([k, n]) => `${k} ${pct(n / total)}`);
	return top.join(", ") + (Object.keys(answers).length > 3 ? ", …" : "");
}

async function addExamples(run, box, btn) {
	btn.disabled = true;
	try {
		const { added } = await api(`/projects/${S.pid}/runs/${run.id}/monitor/examples`, { method: "POST" });
		await loadPage();
		app.renderStepper();
		toast(`Added ${plural(added, "example")} from feedback. Train again to learn from them.`, false, { label: "Open", onclick: () => app.showTab("examples") });
		load(run, box);
	} catch (e) {
		toastError(e);
		btn.disabled = false;
	}
}

function render(run, box, data) {
	const st = data.stats;
	const head = h("div", { class: "card-head" },
		h("div", {}, h("h3", { class: "section-title" }, "In production"),
			h("p", { class: "muted small" }, `Decisions logged by llmzempic serve, ${when(st.first)} to ${when(st.last)}.`)));
	mount(box, head, h("div", { class: "card-body stack" },
		h("div", { class: "metrics" },
			[[st.count.toLocaleString(), "Decisions"], [pct(st.refusal_rate), "Refused (any question)"],
				[st.latency_ms ? `${Math.round(st.latency_ms.p50)} ms` : "–", "Latency (p50)"], [st.feedback.toLocaleString(), "With feedback"]]
				.map(([value, label]) => h("div", { class: "metric" }, h("span", { class: "metric-value num" }, value), h("span", { class: "metric-label" }, label)))),
		h("div", { class: "table-wrap" }, h("table", { class: "decision-table" },
			h("thead", {}, h("tr", {},
				h("th", {}, "Question"),
				h("th", {}, "Answers"),
				h("th", { class: "num", title: "Population stability index of the answers against the training labels; above 0.2 the inputs have likely shifted" }, "Drift"),
				h("th", { class: "num" }, "Confidence"),
				h("th", { class: "num" }, "Refused"),
				h("th", { class: "num", title: "Accuracy on decisions that got feedback" }, "Accuracy"),
			)),
			h("tbody", {}, st.questions.map((q) => h("tr", {},
				h("td", {}, h("code", {}, q.name)),
				h("td", { class: "small" }, distribution(q.answers)),
				h("td", { class: "num" }, q.drift_psi === undefined ? "–" : q.drifted ? pill(fixed(q.drift_psi), "warn", { iconName: "alert" }) : fixed(q.drift_psi)),
				h("td", { class: "num" }, pct(q.mean_confidence)),
				h("td", { class: "num" }, pct(q.refusal_rate)),
				h("td", { class: "num" }, q.checked ? `${pct(q.accuracy)} of ${q.checked}` : "–"),
			))),
		)),
		st.days.length > 1 ? h("details", {},
			h("summary", { class: "small" }, "By day"),
			h("div", { class: "table-wrap" }, h("table", { class: "decision-table" },
				h("thead", {}, h("tr", {}, ["Day", "Decisions", "Confidence", "Refused", "Accuracy"].map((t, i) => h("th", { class: i ? "num" : "" }, t)))),
				h("tbody", {}, st.days.slice().reverse().map((d) => h("tr", {},
					h("td", {}, d.day), h("td", { class: "num" }, String(d.count)), h("td", { class: "num" }, pct(d.mean_confidence)),
					h("td", { class: "num" }, pct(d.refusal_rate)), h("td", { class: "num" }, d.checked ? `${pct(d.accuracy)} of ${d.checked}` : "–"),
				))),
			))) : null,
		data.examples ? h("div", { class: "spread" },
			h("span", { class: "muted small" }, `${plural(data.examples, "decision")} with feedback can become training examples.`),
			button(`Add ${plural(data.examples, "example")}`, { icon: "plus", kind: "secondary", size: "sm", onclick: (e) => addExamples(run, box, e.currentTarget) })) : null,
	));
}

async function load(run, box) {
	try {
		const data = await api(`/projects/${S.pid}/runs/${run.id}/monitor`);
		if (data.stats) render(run, box, data);
		else mount(box);
	} catch (e) {
		mount(box, h("p", { class: "faint small" }, e.message));
	}
}

export function monitorCard(run) {
	const box = h("section", { class: "card monitor-card" }, h("div", { class: "card-body" }, spinnerLine("Reading the decision log…")));
	load(run, box);
	return box;
}
