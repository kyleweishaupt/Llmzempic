import { api, uploadFile } from "./api.js";
import { button, callout, h, mount, toggleSwitch } from "./dom.js";
import { progressDialog, toast, toastError } from "./feedback.js";
import { bytes } from "./format.js";
import { loadProjects } from "./project.js";
import { S, app } from "./state.js";

async function openNew(view, verb) {
	await loadProjects();
	app.renderSidebar();
	await app.openProject(view.id);
	toast(`${verb} "${view.config.name}"`);
}

// One .llmz file with the project's settings, examples, images and data sets, to carry on elsewhere.
export function exportPackageDialog() {
	const hasRun = S.runs.some((r) => r.status.state === "completed");
	let includeRun = false;
	const status = h("div", {});
	const go = button("Export package", { icon: "download", kind: "primary" });
	const dlg = h("dialog", { class: "confirm" });
	const close = () => { dlg.close(); dlg.remove(); };
	mount(dlg,
		h("div", { class: "dialog-head" }, h("div", {},
			h("h2", {}, "Export a package"),
			h("p", {}, "One .llmz file with this project's settings, examples, images and data sets. Import it on another machine to keep curating or to train a better model."))),
		h("div", { class: "dialog-body" },
			toggleSwitch("Include the latest trained run", false, (v) => (includeRun = v),
				{ disabled: !hasRun, title: hasRun ? "Its adapter, scores and logs, not its exports" : "No completed run yet" }),
			status),
		h("div", { class: "dialog-foot" }, button("Cancel", { kind: "ghost", onclick: close }), go),
	);
	go.addEventListener("click", async () => {
		go.disabled = true;
		mount(status, h("div", { class: "row muted small" }, h("span", { class: "spinner" }), "Building the package. Large projects take a while."));
		try {
			const ready = await api(`/projects/${S.pid}/package`, { method: "POST", body: { include_run: includeRun } });
			if (!dlg.open) return;  // cancelled while it was building
			h("a", { href: `/api/packages/${ready.token}`, download: ready.filename }).click();
			toast(`Downloading ${ready.filename} (${bytes(ready.size)})`);
			close();
		} catch (e) {
			mount(status, callout("bad", "alert-circle", e.message));
			go.disabled = false;
		}
	});
	dlg.addEventListener("close", () => dlg.remove());
	document.body.append(dlg);
	dlg.showModal();
}

export async function duplicateProject() {
	const busy = progressDialog(`Duplicating "${S.config.name}"`, "Copying examples, images and settings.");
	try {
		await openNew(await api(`/projects/${S.pid}/duplicate`, { method: "POST" }), "Created");
	} catch (e) {
		toastError(e);
	} finally {
		busy.close();
	}
}

export function importPackage() {
	const input = h("input", { type: "file", accept: ".llmz,.zip" });
	input.addEventListener("change", async () => {
		const file = input.files[0];
		if (!file) return;
		const busy = progressDialog(`Importing ${file.name}`, `Uploading ${bytes(file.size)}…`);
		try {
			const view = await uploadFile("/packages", file, (share) => {
				busy.progress(share);
				if (share >= 1) busy.text("Unpacking the project…");
			});
			await openNew(view, "Imported");
		} catch (e) {
			toastError(e);
		} finally {
			busy.close();
		}
	});
	input.click();
}
