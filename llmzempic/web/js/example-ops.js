import { api } from "./api.js";
import { saver, toast, toastError } from "./feedback.js";
import { plural } from "./format.js";
import { loadPage, pageSize } from "./project.js";
import { S, app } from "./state.js";

export const visibleExamples = () => S.examples;

export const currentFilter = () => ({ kind: S.query.filter, search: S.query.search, label: S.query.label });

// What a bulk action applies to: the ticked rows, or every example matching the query.
export const selectionScope = () => (S.selectAll ? { filter: currentFilter() } : { ids: [...S.selection] });

export async function reloadExamples() {
	await loadPage();
	if (!S.examples.length && S.query.offset > 0) {
		S.query.offset = Math.max(0, S.query.offset - pageSize());
		await loadPage();
	}
	app.renderExamples({ keepScroll: true });
	app.renderStepper();
}

// Bulk changes to 200k examples take seconds; the selection bar says what's happening meanwhile.
async function working(text, task) {
	S.busy = text;
	app.renderExamples({ keepScroll: true });
	try {
		await task();
	} catch (e) {
		toastError(e);
	} finally {
		S.busy = null;
		app.renderExamples({ keepScroll: true });
	}
}

async function bulk(body, doing, describe) {
	await saver.flush();
	await working(doing, async () => {
		const res = await api(`/projects/${S.pid}/examples/bulk`, { method: "POST", body });
		S.selection.clear();
		S.selectAll = false;
		await reloadExamples();
		toast(describe(res.affected), false, {
			label: "Undo",
			onclick: () => working("Undoing…", async () => {
				await api(`/projects/${S.pid}/examples/undo`, { method: "POST", body: { op: res.undo } });
				await reloadExamples();
			}),
		});
	});
}

// Delete with an Undo toast instead of a confirmation: curating thousands of examples needs to be quick.
export const deleteExamples = (scope) => bulk({ op: "delete", ...scope }, "Deleting…", (n) => `Deleted ${plural(n, "example")}.`);

export const setAnswer = (scope, output) => bulk({ op: "set_output", output, ...scope }, "Setting the answer…", (n) => `Set the answer for ${plural(n, "example")}.`);

export async function addExamples(rows) {
	return (await api(`/projects/${S.pid}/examples`, { method: "POST", body: { examples: rows } })).examples;
}
