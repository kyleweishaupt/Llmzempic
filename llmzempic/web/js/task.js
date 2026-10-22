import { api } from "./api.js";
import { $, button, callout, field, h, icon, menu, mount, seg, toggleSwitch } from "./dom.js";
import { confirmAction, saver, toast, toastError } from "./feedback.js";
import { jsonBlock } from "./json-view.js";
import { scheduleConfigSave } from "./project.js";
import { fromSchema, sample } from "./schema-model.js";
import { schemaEditor } from "./schema-editor.js";
import { S, app, catalogModel, recommendedModel } from "./state.js";

const KIND_CARDS = [
	["text", "text", "Text", "Messages, documents, file contents."],
	["image", "image", "Images", "Photos or screenshots, with optional text."],
];

function pane() { return $("#tab-task"); }

function previewBody() {
	const parsed = fromSchema(S.config.schema);
	if (!parsed.ok) return h("p", { class: "faint" }, "A preview appears for schemas the visual editor understands.");
	if (!parsed.root.fields.length) return h("p", { class: "faint" }, "Add output fields to see the answer take shape.");
	return jsonBlock(sample(parsed.root));
}

function updatePreview() {
	const slot = $("#answer-preview");
	if (slot) mount(slot, previewBody());
}

function updateServerBits() {
	const prompt = $("#system-prompt");
	if (prompt) prompt.textContent = S.view.system_prompt;
	const slot = $("#schema-errors");
	if (slot) mount(slot, (S.view.schema_errors || []).map((e) => callout("bad", "alert-circle", e)));
}

// Switching the input kind also moves the base model to one that can read that kind of input.
async function setInputKind(kind) {
	if (kind === S.config.input_kind) return;
	if (kind === "image" && S.examples.some((e) => !e.image)) {
		const ok = await confirmAction({
			title: "Switch this project to images?",
			text: "Examples without an image will be flagged until you give each one a picture or delete it.",
			action: "Switch to images",
		});
		if (!ok) return;
	}
	S.config.input_kind = kind;
	const entry = catalogModel(S.config.base_model);
	if (entry && Boolean(entry.vision) !== (kind === "image")) {
		const next = recommendedModel(kind);
		if (next) {
			S.config.base_model = next;
			toast(`Base model switched to ${catalogModel(next).name}, which can ${kind === "image" ? "read images" : "work with text"}.`);
		}
	}
	scheduleConfigSave();
	app.renderProjectHead();
	renderTask();
}

async function applyTemplate(t) {
	const ok = await confirmAction({
		title: `Use the "${t.name}" template?`,
		text: "It replaces the instructions, the output structure and the input type.",
		action: "Use template",
	});
	if (!ok) return;
	S.config.instructions = t.instructions;
	S.config.schema = structuredClone(t.schema);
	S.schemaMode = "visual";
	const kind = t.input_kind || "text";
	if (kind !== S.config.input_kind) await setInputKind(kind);
	scheduleConfigSave();
	renderTask();
}

async function inferSchema() {
	try {
		await saver.flush();
		const { schema } = await api(`/projects/${S.pid}/infer-schema`, { method: "POST" });
		S.config.schema = schema;
		scheduleConfigSave();
		renderTask();
		toast("Built from your examples' answers. Check the option lists and required fields.");
	} catch (e) {
		toastError(e);
	}
}

function jsonEditor() {
	const status = h("div", { class: "json-status" });
	const area = h("textarea", { class: "code schema-json", spellcheck: "false", rows: 18, value: JSON.stringify(S.config.schema, null, 2), "aria-label": "Output JSON Schema" });
	area.addEventListener("input", () => {
		try {
			const schema = JSON.parse(area.value);
			if (!schema || typeof schema !== "object" || Array.isArray(schema)) throw new Error("The schema must be a JSON object.");
			S.config.schema = schema;
			area.classList.remove("invalid");
			mount(status);
			scheduleConfigSave();
			updatePreview();
		} catch (err) {
			area.classList.add("invalid");
			mount(status, callout("warn", "alert", `Not valid JSON yet: ${err.message}`));
		}
	});
	return h("div", { class: "stack" }, area, status);
}

function outputCard() {
	const parsed = fromSchema(S.config.schema);
	const mode = parsed.ok ? S.schemaMode : "json";
	const body = mode === "visual"
		? schemaEditor(parsed.root, (schema) => { S.config.schema = schema; scheduleConfigSave(); updatePreview(); })
		: jsonEditor();
	const templates = S.catalog.templates;
	return h("section", { class: "card" },
		h("div", { class: "card-head" },
			h("div", {},
				h("h2", { class: "section-title" }, "Output"),
				h("p", { class: "muted small" }, "The JSON object the model answers with. Every example is checked against it."),
			),
			h("div", { class: "row" },
				seg([["visual", "Visual"], ["json", "JSON Schema"]], mode, (m) => { S.schemaMode = m; renderTask(); }, { label: "Editor", disabled: !parsed.ok }),
				menu(button("Start from…", { icon: "layers", kind: "secondary", size: "sm" }), [
					...templates.map((t) => ({ label: t.name, icon: t.input_kind === "image" ? "image" : "text", onclick: () => applyTemplate(t) })),
					"-",
					{ label: "Infer from example answers", icon: "braces", disabled: !S.examples.length, onclick: inferSchema },
				]),
			),
		),
		h("div", { class: "card-body stack" },
			parsed.ok ? null : callout("info", "info", h("b", {}, "Edit as JSON. "), `${parsed.reason} The visual editor covers text, numbers, yes/no, option lists, lists and nested groups.`),
			body,
			h("div", { id: "schema-errors", class: "stack" }),
			toggleSwitch("Include the schema in the system prompt (recommended)", S.config.include_schema_in_prompt !== false, (v) => {
				S.config.include_schema_in_prompt = v;
				scheduleConfigSave();
			}),
		),
	);
}

function basicsCard() {
	const name = h("input", { type: "text", value: S.config.name || "", "aria-label": "Project name" });
	name.addEventListener("input", () => {
		S.config.name = name.value;
		$("#project-title").textContent = name.value || S.pid;
		scheduleConfigSave();
	});
	const instructions = h("textarea", { rows: 4, value: S.config.instructions || "", placeholder: "Decide whether the image shows or contains a hot dog." });
	instructions.addEventListener("input", () => {
		S.config.instructions = instructions.value;
		scheduleConfigSave();
	});
	const kind = S.config.input_kind || "text";
	return h("section", { class: "card card-pad stack-lg" },
		field("Name", name),
		h("div", { class: "field" },
			h("span", { class: "field-label" }, "Input"),
			h("div", { class: "choice-cards", role: "radiogroup", "aria-label": "Input type" }, KIND_CARDS.map(([value, iconName, title, text]) => h("button", {
				type: "button", role: "radio", class: "choice-card", "aria-checked": String(kind === value), onclick: () => setInputKind(value),
			}, h("span", { class: "choice-icon" }, icon(iconName)), h("span", {}, h("b", {}, title), h("span", { class: "muted small" }, text)),
				kind === value ? icon("check-circle", "choice-check") : null))),
		),
		field("Instructions", instructions, { help: "Tell the model what to decide. This becomes its system prompt, so be specific about edge cases." }),
	);
}

function asideCard() {
	const copy = button(null, { icon: "copy", kind: "ghost", size: "sm", title: "Copy system prompt", onclick: async (e) => {
		e.preventDefault(); // inside <summary>: copy without toggling
		await navigator.clipboard.writeText(S.view.system_prompt);
		toast("System prompt copied");
	} });
	return h("aside", { class: "task-aside" },
		h("section", { class: "card preview-card" },
			h("div", { class: "card-head" }, h("h2", { class: "section-title" }, "Example answer")),
			h("div", { class: "card-body" },
				h("div", { id: "answer-preview" }, previewBody()),
				h("p", { class: "faint small" }, "Placeholder values in the shape your model will return."),
			),
		),
		h("details", { class: "card prompt-card" },
			h("summary", { class: "card-head" }, h("span", { class: "section-title" }, "System prompt"), copy),
			h("div", { class: "card-body" }, h("pre", { id: "system-prompt", class: "code-block scroll small" }, S.view.system_prompt)),
		),
	);
}

export function renderTask() {
	mount(pane(), h("div", { class: "task-grid" },
		h("div", { class: "stack-lg" }, basicsCard(), outputCard()),
		asideCard(),
	));
	updateServerBits();
}

app.configSaved = () => { if (S.tab === "task") updateServerBits(); };
