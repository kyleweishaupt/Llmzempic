import { $, callout, field, h, icon, mount, pill } from "./dom.js";
import { scheduleConfigSave } from "./project.js";
import { S, app, catalogModel, hardware, isImageProject, modelsForKind } from "./state.js";

const FIT = {
	good: ["good", "Fits"],
	tight: ["warn", "Tight fit"],
	slow: ["warn", "Slow here"],
	"too-big": ["bad", "Too big"],
};

const PRESETS = {
	quick: ["Quick", "A first look in minutes. Good while you're still collecting examples."],
	balanced: ["Balanced", "The usual choice once your examples are in good shape."],
	thorough: ["Thorough", "Learns the most from every example. Slowest."],
};

const TRAINING_FIELDS = [
	{ key: "epochs", label: "Epochs", step: 1, min: 1, help: "Passes over the examples." },
	{ key: "learning_rate", label: "Learning rate", step: 0.00001, min: 0 },
	{ key: "lora_r", label: "LoRA rank", step: 1, min: 1, help: "Higher learns more and uses more memory." },
	{ key: "lora_alpha", label: "LoRA alpha", step: 1, min: 1 },
	{ key: "lora_dropout", label: "LoRA dropout", step: 0.01, min: 0 },
	{ key: "batch_size", label: "Batch size", step: 1, min: 1, help: "Lower it if training runs out of memory." },
	{ key: "grad_accum", label: "Gradient accumulation", step: 1, min: 1 },
	{ key: "max_seq_length", label: "Max tokens per example", step: 64, min: 0, help: "0 fits the longest example." },
	{ key: "eval_fraction", label: "Held-out share", step: 0.05, min: 0, help: "Kept aside for scoring (needs 10+ examples)." },
	{ key: "quantization", label: "Quantization", options: [["auto", "Auto"], ["4bit", "4-bit QLoRA"], ["none", "None (16-bit)"]] },
	{ key: "accelerator", label: "Accelerator", options: [["auto", "Auto (Unsloth if installed)"], ["unsloth", "Unsloth"], ["off", "Standard PEFT"]], help: "Unsloth is used for text models only." },
	{ key: "seed", label: "Random seed", step: 1, min: 0 },
];

function selectModel(id) {
	S.config.base_model = id;
	scheduleConfigSave();
	renderModel();
}

function hardwareNote() {
	const hw = hardware();
	if (!hw) return null;
	if (hw.device !== "cuda") {
		return callout("warn", "cpu", h("b", {}, "No NVIDIA GPU found. "), "Training runs on the CPU: fine for the smallest models, very slow above 1B parameters. Exported models still run well on CPUs.");
	}
	const parts = [`${hw.gpus[0].name} with ${hw.vram_gb} GB of video memory.`];
	parts.push(hw.can_4bit ? "Models that don't fit at 16-bit train in 4-bit automatically." : "Install bitsandbytes to train bigger models in 4-bit.");
	if (!isImageProject()) parts.push(hw.can_unsloth ? "Unsloth makes training faster." : "");
	return h("p", { class: "muted" }, parts.join(" "));
}

function modelCard(m) {
	const selected = m.id === S.config.base_model;
	const [tone, fitText] = FIT[m.fit] || ["", m.fit];
	return h("button", { type: "button", class: "model-card", role: "radio", "aria-checked": String(selected), onclick: () => selectModel(m.id) },
		h("div", { class: "spread" },
			h("span", { class: "faint small" }, m.tier),
			selected ? icon("check-circle", "model-check") : m.recommended ? pill("Recommended", "info") : null,
		),
		h("span", { class: "model-name" }, m.name),
		h("code", { class: "model-id ellipsis" }, m.id),
		h("div", { class: "model-facts" },
			h("span", { class: "model-size num" }, `${m.params_b < 1 ? Math.round(m.params_b * 1000) + "M" : m.params_b + "B"}`),
			h("span", { class: "faint" }, m.license),
			m.gated ? pill("Gated", "warn", { iconName: "alert" }) : null,
		),
		h("div", { class: "model-fit" }, pill(fitText, tone, { dot: true }), h("span", { class: "faint small" }, m.fit_detail)),
		h("p", { class: "muted small" }, m.notes),
	);
}

function modelSection() {
	const kind = isImageProject() ? "image" : "text";
	const models = modelsForKind(kind);
	const current = catalogModel(S.config.base_model);
	const custom = h("input", { type: "text", class: "code", value: current ? "" : S.config.base_model || "", placeholder: kind === "image" ? "e.g. Qwen/Qwen3-VL-8B-Instruct" : "e.g. Qwen/Qwen2.5-7B-Instruct", spellcheck: "false" });
	custom.addEventListener("change", () => { if (custom.value.trim()) selectModel(custom.value.trim()); });
	const mismatch = current && Boolean(current.vision) !== (kind === "image");
	return h("section", { class: "stack-lg" },
		h("div", { class: "stack" },
			h("h2", { class: "section-title" }, "Base model", h("span", { class: "faint" }, kind === "image" ? "vision models read images" : "text models")),
			hardwareNote(),
		),
		mismatch ? callout("bad", "alert-circle", h("b", {}, `${current.name} ${kind === "image" ? "can't read images." : "is a vision model."} `), `Pick one of the ${kind === "image" ? "vision" : "text"} models below.`) : null,
		models.length
			? h("div", { class: "model-grid", role: "radiogroup", "aria-label": "Base model" }, models.map(modelCard))
			: callout("info", "info", "No catalog models for this input type yet. Enter a model id below."),
		h("div", { class: "custom-model" },
			field("Another model", custom, { hint: "any Hugging Face id or local folder", help: current ? null : S.config.base_model ? `Using ${S.config.base_model}.` : null })),
	);
}

function presetSection() {
	const t = S.config.training;
	const active = Object.entries(S.catalog.presets).find(([, values]) => Object.entries(values).every(([k, v]) => Number(t[k]) === v));
	const activeName = active ? active[0] : "custom";
	const controls = TRAINING_FIELDS.map((f) => {
		const value = t[f.key];
		const control = f.options
			? h("select", {}, f.options.map(([v, label]) => h("option", { value: v, selected: v === value }, label)))
			: h("input", { type: "number", step: f.step, min: f.min, value });
		control.addEventListener("change", () => {
			t[f.key] = f.options ? control.value : Number(control.value);
			scheduleConfigSave();
			mount($("#preset-cards"), presetCards());
		});
		return field(f.label, control, { help: f.help });
	});
	const presetCards = () => {
		const cur = Object.entries(S.catalog.presets).find(([, values]) => Object.entries(values).every(([k, v]) => Number(t[k]) === v));
		return Object.entries(S.catalog.presets).map(([name, values]) => {
			const [title, text] = PRESETS[name] || [name, ""];
			return h("button", {
				type: "button", class: "preset-card", role: "radio", "aria-checked": String(cur && cur[0] === name),
				onclick: () => {
					Object.assign(t, values, { preset: name });
					scheduleConfigSave();
					renderModel();
				},
			}, h("b", {}, title), h("span", { class: "muted small" }, text),
				h("span", { class: "faint small" }, `${values.epochs} epochs, LoRA rank ${values.lora_r}`));
		});
	};
	return h("section", { class: "stack-lg" },
		h("h2", { class: "section-title" }, "Training", activeName === "custom" ? h("span", { class: "faint" }, "custom settings") : null),
		h("div", { id: "preset-cards", class: "preset-grid", role: "radiogroup", "aria-label": "Training preset" }, presetCards()),
		h("details", { class: "card advanced" },
			h("summary", {}, icon("sliders"), "Advanced settings"),
			h("div", { class: "settings-grid" }, controls),
		),
	);
}

export function renderModel() {
	mount($("#tab-model"), h("div", { class: "stack-lg model-page" }, modelSection(), presetSection()));
	app.renderStepper();
}
