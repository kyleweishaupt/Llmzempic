import { toast } from "./feedback.js";
import { plural } from "./format.js";
import { matchesLabel } from "./health.js";
import { scheduleExamplesSave } from "./project.js";
import { S, app, duplicateCounts, exampleErrors } from "./state.js";

export function visibleExamples() {
	let list = S.examples;
	if (S.filter === "problems") list = list.filter((e) => exampleErrors(e.id).length);
	if (S.filter === "duplicates") {
		// Copies of the same input are listed next to each other.
		const byId = new Map(list.map((e) => [e.id, e]));
		list = ((S.stats && S.stats.duplicates) || []).flatMap((g) => g.ids).map((id) => byId.get(id)).filter(Boolean);
	}
	if (S.labelFilter) list = list.filter((e) => matchesLabel(e, S.labelFilter));
	const q = S.search.trim().toLowerCase();
	if (q) list = list.filter((e) => e.input.toLowerCase().includes(q) || JSON.stringify(e.output ?? "").toLowerCase().includes(q));
	return list;
}

export const filterCounts = () => ({
	all: S.examples.length,
	problems: S.examples.filter((e) => exampleErrors(e.id).length).length,
	duplicates: duplicateCounts().ids.size,
});

// Delete with an Undo toast instead of a confirmation: curating hundreds of images needs to be quick.
export function deleteExamples(ids) {
	const gone = new Set(ids);
	const removed = S.examples.map((e, i) => [i, e]).filter(([, e]) => gone.has(e.id));
	if (!removed.length) return;
	S.examples = S.examples.filter((e) => !gone.has(e.id));
	for (const id of gone) {
		S.selection.delete(id);
		S.badJson.delete(id);
	}
	scheduleExamplesSave();
	app.renderExamples();
	toast(`Deleted ${plural(removed.length, "example")}.`, false, {
		label: "Undo",
		onclick: () => {
			for (const [i, e] of removed) S.examples.splice(Math.min(i, S.examples.length), 0, e);
			scheduleExamplesSave();
			app.renderExamples();
		},
	});
}

export function setAnswer(ids, output) {
	const target = new Set(ids);
	const before = new Map();
	for (const e of S.examples) {
		if (!target.has(e.id)) continue;
		before.set(e.id, e.output);
		e.output = structuredClone(output);
	}
	scheduleExamplesSave();
	app.renderExamples();
	toast(`Set the answer for ${plural(before.size, "example")}.`, false, {
		label: "Undo",
		onclick: () => {
			for (const e of S.examples) if (before.has(e.id)) e.output = before.get(e.id);
			scheduleExamplesSave();
			app.renderExamples();
		},
	});
}
