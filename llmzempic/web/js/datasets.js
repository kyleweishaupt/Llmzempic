import { api, uploadFile } from "./api.js";
import { $, button, callout, emptyState, field, h, icon, mount, spinnerLine } from "./dom.js";
import { confirmAction, toast, toastError } from "./feedback.js";
import { bytes, plural, when } from "./format.js";
import { S } from "./state.js";

const PREVIEW_BYTES = 64 * 1024;

export async function loadDatasets() {
	S.datasets = (await api("/datasets")).datasets;
	return S.datasets;
}

function pickFile(onFile) {
	const input = h("input", { type: "file", accept: ".csv,.tsv,.txt,.json" });
	input.addEventListener("change", () => input.files[0] && onFile(input.files[0]));
	input.click();
}

// Upload (or replace) a data set: preview the start of the file, choose columns, then stream it.
async function uploadDialog(file, replacing = null) {
	const head = await file.slice(0, PREVIEW_BYTES).text();
	let info;
	try {
		info = await api("/datasets/preview", { method: "POST", body: { text: head, filename: file.name } });
	} catch (e) {
		toastError(e);
		return;
	}
	const dlg = h("dialog", { class: "wide dataset-upload" });
	const name = h("input", { type: "text", value: replacing ? replacing.name : file.name.replace(/\.[^.]+$/, ""), disabled: Boolean(replacing) });
	const valueCol = info.columns ? h("select", {}, info.columns.map((c) => h("option", { value: c, selected: c === info.value_column }, c))) : null;
	const descCol = info.columns ? h("select", {},
		h("option", { value: "", selected: !info.description_column }, "None"),
		info.columns.map((c) => h("option", { value: c, selected: c === info.description_column }, c))) : null;
	const previewSlot = h("div", {});
	const progress = h("div", { class: "stack", hidden: true });
	const go = button(replacing ? "Replace values" : "Add data set", { kind: "primary", icon: "upload" });
	const cancel = button("Cancel", { kind: "ghost", onclick: () => dlg.close() });

	const drawPreview = (rows) => mount(previewSlot, rows.length
		? h("div", { class: "table-wrap" }, h("table", { class: "dataset-preview" },
			h("thead", {}, h("tr", {}, h("th", {}, "Value"), h("th", {}, "Description"))),
			h("tbody", {}, rows.slice(0, 8).map((r) => h("tr", {}, h("td", {}, h("code", {}, r.value)), h("td", { class: "muted" }, r.description || "–"))))))
		: callout("warn", "alert", "No values found at the start of this file."));
	drawPreview(info.rows);
	for (const select of [valueCol, descCol].filter(Boolean)) {
		select.addEventListener("change", async () => {
			if (select === valueCol && descCol.value === valueCol.value) descCol.value = "";
			const again = await api("/datasets/preview", { method: "POST", body: { text: head, filename: file.name, value_column: valueCol.value, description_column: descCol.value } });
			drawPreview(again.rows);
		});
	}

	go.addEventListener("click", async () => {
		go.disabled = true;
		cancel.disabled = true;
		const bar = h("span", { style: { width: "0%" } });
		const label = h("span", { class: "muted small" }, "Uploading…");
		mount(progress, h("div", { class: "progress" }, bar), label);
		progress.hidden = false;
		const params = new URLSearchParams({ filename: file.name });
		if (!replacing) params.set("name", name.value.trim() || file.name);
		if (valueCol) params.set("value_column", valueCol.value);
		if (descCol) params.set("description_column", descCol.value);
		try {
			const meta = await uploadFile(replacing ? `/datasets/${replacing.id}?${params}` : `/datasets?${params}`, file, (share) => {
				bar.style.width = `${Math.round(share * 100)}%`;
				if (share >= 1) {
					label.textContent = "Building the search index…";
					progress.firstChild.classList.add("indeterminate");
				}
			}, replacing ? "PUT" : "POST");
			dlg.close();
			toast(meta.reused ? `That list is already in the library as "${meta.name}".` : `${replacing ? "Replaced" : "Added"} "${meta.name}" with ${plural(meta.rows, "value")}.`);
			renderDatasets();
		} catch (e) {
			mount(progress, callout("bad", "alert-circle", e.message));
			go.disabled = false;
			cancel.disabled = false;
		}
	});

	mount(dlg,
		h("div", { class: "dialog-head" }, h("div", {},
			h("h2", {}, replacing ? `Replace "${replacing.name}"` : "Add a data set"),
			h("p", {}, `${file.name}, ${bytes(file.size)}. `, replacing ? "Projects using it are checked again; past runs keep the values they trained with." : "Values are matched exactly, including case."))),
		h("div", { class: "dialog-body" },
			replacing ? null : field("Name", name),
			info.columns ? h("div", { class: "row" }, field("Values", valueCol), field("Descriptions", descCol, { hint: "optional" })) : null,
			h("div", { class: "field" }, h("span", { class: "field-label" }, "First rows"), previewSlot),
			progress,
		),
		h("div", { class: "dialog-foot" }, cancel, go),
	);
	dlg.addEventListener("close", () => dlg.remove());
	document.body.append(dlg);
	dlg.showModal();
}

function searchPanel(d) {
	const results = h("div", {});
	const input = h("input", { type: "search", placeholder: `Search ${plural(d.rows, "value")}`, "aria-label": `Search ${d.name}` });
	let timer;
	const run = async () => {
		try {
			const { rows } = await api(`/datasets/${encodeURIComponent(d.id)}/search?q=${encodeURIComponent(input.value)}&limit=25`);
			mount(results, rows.length
				? h("div", { class: "table-wrap" }, h("table", {}, h("tbody", {}, rows.map((r) => h("tr", {}, h("td", {}, h("code", {}, r.value)), h("td", { class: "muted" }, r.description || ""))))))
				: h("p", { class: "faint" }, "No matching values."));
		} catch (e) {
			mount(results, callout("bad", "alert-circle", e.message));
		}
	};
	input.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(run, 150); });
	run();
	return h("div", { class: "stack dataset-search" }, h("div", { class: "search-box" }, icon("search"), input), results);
}

function datasetCard(d) {
	const slot = h("div", {});
	const remove = async () => {
		if (d.used_by.length) {
			toast(`"${d.name}" is used by ${d.used_by.join(", ")}. Remove it from their schemas first.`, true);
			return;
		}
		const ok = await confirmAction({ title: `Delete "${d.name}"?`, text: "Runs that used it keep their own copy.", action: "Delete data set", danger: true });
		if (!ok) return;
		try {
			await api(`/datasets/${encodeURIComponent(d.id)}`, { method: "DELETE" });
			toast(`Deleted "${d.name}".`);
			renderDatasets();
		} catch (e) {
			toastError(e);
		}
	};
	return h("section", { class: "card dataset-card" },
		h("div", { class: "spread" },
			h("div", { class: "row" },
				h("span", { class: "project-icon" }, icon("database")),
				h("div", {},
					h("h3", {}, d.name),
					h("span", { class: "faint small" }, `${plural(d.rows, "value")}, from ${d.source}`, d.updated || d.created ? `, ${when(d.updated || d.created)}` : ""),
				),
			),
			h("div", { class: "row" },
				button("Search", { icon: "search", kind: "ghost", size: "sm", onclick: () => mount(slot, slot.firstChild ? null : searchPanel(d)) }),
				button("Replace…", { icon: "upload", kind: "ghost", size: "sm", onclick: () => pickFile((file) => uploadDialog(file, d)) }),
				button(null, { icon: "trash", kind: "danger-quiet", size: "sm", title: `Delete ${d.name}`, onclick: remove }),
			),
		),
		h("div", { class: "row small" },
			h("code", { class: "faint" }, d.id),
			d.used_by.length ? [h("span", { class: "muted" }, "Used by"), d.used_by.map((n) => h("span", { class: "chip" }, n))] : h("span", { class: "faint" }, "Not used by any project yet"),
		),
		slot,
	);
}

export async function renderDatasets() {
	const pane = $("#datasets-page");
	mount(pane, spinnerLine("Loading data sets…"));
	try {
		await loadDatasets();
	} catch (e) {
		mount(pane, callout("bad", "alert-circle", e.message));
		return;
	}
	mount(pane, h("div", { class: "datasets-page stack-lg" },
		h("div", { class: "spread" },
			h("div", { class: "stack" },
				h("h1", {}, "Data sets"),
				h("p", { class: "lede" }, "Lists of allowed values an output field can draw from, shared by every project. Answers are checked against them, and exported models served by llama-server can only answer with their values."),
			),
			button("Add a data set", { kind: "primary", icon: "upload", onclick: () => pickFile((file) => uploadDialog(file)) }),
		),
		S.datasets.length
			? h("div", { class: "stack" }, S.datasets.map(datasetCard))
			: emptyState("database", "No data sets yet", "Upload a CSV, TSV, TXT or JSON file of values, with optional descriptions. Large lists (a million values) are fine.",
				button("Add a data set", { kind: "primary", icon: "upload", onclick: () => pickFile((file) => uploadDialog(file)) })),
	));
}
