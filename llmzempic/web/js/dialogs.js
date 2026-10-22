import { api } from "./api.js";
import { button, field, h, icon, mount } from "./dom.js";
import { toast, toastError } from "./feedback.js";
import { loadProjects } from "./project.js";
import { S, app } from "./state.js";

function modal(build) {
	const dlg = h("dialog", {});
	const close = () => { dlg.close(); };
	dlg.addEventListener("close", () => dlg.remove());
	mount(dlg, build(close));
	document.body.append(dlg);
	dlg.showModal();
	return dlg;
}

export function newProjectDialog() {
	let template = "classification";
	const name = h("input", { type: "text", required: true, placeholder: "Is it a hot dog?", "aria-label": "Project name" });
	const grid = h("div", { class: "template-grid", role: "radiogroup", "aria-label": "Starting point" });
	const draw = () => mount(grid, S.catalog.templates.map((t) => h("button", {
		type: "button", role: "radio", class: "template-card", "aria-checked": String(t.id === template),
		onclick: () => { template = t.id; draw(); },
	},
		h("span", { class: "choice-icon" }, icon(t.input_kind === "image" ? "image" : "text")),
		h("span", {}, h("b", {}, t.name), h("span", { class: "muted small" }, t.description || "")),
	)));
	draw();
	const dlg = modal((close) => h("form", { method: "dialog", onsubmit: async (e) => {
		e.preventDefault();
		if (!name.value.trim()) { name.focus(); return; }
		const submit = e.submitter || dlg.querySelector("[type=submit]");
		submit.disabled = true;
		try {
			const view = await api("/projects", { method: "POST", body: { name: name.value.trim(), template } });
			close();
			await loadProjects();
			app.renderSidebar();
			await app.openProject(view.id, "task");
		} catch (err) {
			toastError(err);
			submit.disabled = false;
		}
	} },
		h("div", { class: "dialog-head" }, h("div", {}, h("h2", {}, "New project"), h("p", {}, "You can change everything later."))),
		h("div", { class: "dialog-body" },
			field("Name", name),
			h("div", { class: "field" }, h("span", { class: "field-label" }, "Start from"), grid),
		),
		h("div", { class: "dialog-foot" },
			button("Cancel", { kind: "ghost", onclick: close }),
			button("Create project", { kind: "primary", type: "submit" }),
		),
	));
	dlg.classList.add("wide");
	name.focus();
}

export async function settingsDialog() {
	let data;
	try {
		data = await api("/settings");
	} catch (e) {
		toastError(e);
		return;
	}
	const token = h("input", { type: "password", autocomplete: "off", placeholder: data.hf_token_set ? "Saved. Type to replace it." : "hf_…" });
	const llama = h("input", { type: "text", class: "code", spellcheck: "false", value: data.llama_cpp_dir || "" });
	modal((close) => h("form", { method: "dialog", onsubmit: async (e) => {
		e.preventDefault();
		const body = { llama_cpp_dir: llama.value.trim() };
		if (token.value.trim()) body.hf_token = token.value.trim();
		try {
			await api("/settings", { method: "PUT", body });
			close();
			toast("Settings saved");
		} catch (err) {
			toastError(err);
		}
	} },
		h("div", { class: "dialog-head" }, h("div", {}, h("h2", {}, "Settings"))),
		h("div", { class: "dialog-body" },
			field("Hugging Face token", token, { help: "Needed for gated models such as Llama and Gemma. Stored in your workspace, never shown again." }),
			field("llama.cpp folder", llama, { hint: "optional", help: "Downloaded automatically on first export when empty." }),
			h("p", { class: "faint small" }, "Workspace: ", h("code", {}, S.system ? S.system.workspace : "")),
		),
		h("div", { class: "dialog-foot" },
			button("Cancel", { kind: "ghost", onclick: close }),
			button("Save", { kind: "primary", type: "submit" }),
		),
	));
}
