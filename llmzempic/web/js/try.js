import { api, runImageUrl } from "./api.js";
import { button, callout, h, icon, mount, pill } from "./dom.js";
import { toast, toastError } from "./feedback.js";
import { imageFiles, uploadPrepared } from "./intake.js";
import { jsonBlock } from "./json-view.js";
import { addExamples } from "./example-ops.js";
import { loadPage } from "./project.js";
import { blankOutput } from "./schema-model.js";
import { pct } from "./format.js";
import { S, activeJob, app } from "./state.js";

const JOB_LABEL = { train: "training", export: "export", predict: "try" };
const vision = (run) => run.input_kind === "image";

async function setImage(files, rerender) {
	const [file] = imageFiles(files);
	if (!file) { toast("Choose an image file.", true); return; }
	S.tryImage = { uploading: true, preview: URL.createObjectURL(file) };
	rerender();
	try {
		S.tryImage = { name: await uploadPrepared(file), preview: S.tryImage.preview };
	} catch (e) {
		S.tryImage = null;
		toastError(e);
	}
	rerender();
}

function imagePicker(rerender) {
	const pick = () => {
		const input = h("input", { type: "file", accept: "image/*" });
		input.addEventListener("change", () => setImage([...input.files], rerender));
		input.click();
	};
	const img = S.tryImage;
	const zone = h("button", { type: "button", class: img ? "dropzone try-zone has-image" : "dropzone try-zone", onclick: pick },
		img ? h("img", { src: img.preview, alt: "Image to try" }) : null,
		img && img.uploading ? h("span", { class: "try-uploading" }, h("span", { class: "spinner" }), "Uploading…") : null,
		img ? null : icon("image"),
		img ? null : h("b", {}, "Drop an image or click to choose"),
		img ? null : h("span", { class: "muted small" }, "Use one the model hasn't seen."),
	);
	zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
	zone.addEventListener("dragleave", () => zone.classList.remove("over"));
	zone.addEventListener("drop", (e) => {
		e.preventDefault();
		e.stopPropagation();
		zone.classList.remove("over");
		setImage([...e.dataTransfer.files], rerender);
	});
	return zone;
}

async function runTry(run) {
	const images = vision(run) ? [S.tryImage && S.tryImage.name] : undefined;
	if (vision(run) ? !images[0] : !S.tryInput.trim()) {
		toast(vision(run) ? "Choose an image to try." : "Enter an input to try.", true);
		return;
	}
	try {
		await api(`/projects/${S.pid}/runs/${run.id}/predict`, { method: "POST", body: { inputs: [S.tryInput], images } });
		await app.refreshSystem();
		app.schedulePoll(800);
	} catch (e) {
		toastError(e);
	}
}

async function addAsExample(it, ok) {
	try {
		await addExamples([{ input: it.input || "", ...(it.image ? { image: it.image } : {}), output: ok ? structuredClone(it.value) : blankOutput(S.config.schema) }]);
	} catch (e) {
		toastError(e);
		return;
	}
	S.query = { filter: "all", search: "", label: null, offset: 0 };
	await loadPage();
	app.renderStepper();
	toast("Added as an example. Check its answer on the Examples tab, then train again.", false, { label: "Open", onclick: () => app.showTab("examples") });
}

const MAX_SHOWN = 8;

function decisionAnswers(response) {
	return h("div", { class: "decision-answers" },
		response.answers.map((a) => {
			let rows = a.type === "predicate" ? [["Yes", a.probability], ["No", 1 - a.probability]] : Object.entries(a.probabilities);
			const top = rows.reduce((best, row) => (row[1] > best[1] ? row : best), rows[0])[0];
			if (a.type === "choice") rows = rows.sort((x, y) => y[1] - x[1]);
			const hidden = rows.length - MAX_SHOWN;
			return h("div", { class: "decision" },
				h("div", { class: "spread" },
					h("span", { class: "row-tight" }, h("code", {}, a.name), h("span", { class: "faint small" }, a.type)),
					h("span", { class: "muted small" }, a.type === "score" && a.score !== null ? `score ${a.score.toFixed(2)}, ` : "", `confidence ${pct(a.confidence)}`),
				),
				rows.slice(0, MAX_SHOWN).map(([label, p]) => h("div", { class: label === top ? "prob-row top" : "prob-row" },
					h("span", { class: "prob-label ellipsis", title: label }, label),
					h("span", { class: "rate-bar" }, h("span", { style: { width: `${p * 100}%` } })),
					h("span", { class: "num" }, pct(p)),
				)),
				hidden > 0 ? h("span", { class: "faint small" }, `…and ${hidden} more options`) : null,
			);
		}),
		h("p", { class: "faint small" }, `${response.usage.input_tokens} input tokens read, nothing generated.`),
	);
}

function result(run, it) {
	const ok = it.json_valid && !it.errors.length;
	return h("div", { class: "try-result" },
		it.image ? h("img", { class: "try-thumb", src: runImageUrl(S.pid, run.id, it.image), alt: "" }) : null,
		h("div", { class: "stack grow" },
			it.input ? h("p", { class: "muted small" }, it.input) : null,
			it.decisions ? decisionAnswers(it.decisions) : it.json_valid ? jsonBlock(it.value) : h("pre", { class: "code-block" }, it.generated || "(no answer)"),
			h("div", { class: "spread" },
				ok ? pill("Matches the schema", "good", { iconName: "check" }) : pill(it.errors[0], it.json_valid ? "warn" : "bad", { iconName: "alert" }),
				button(ok ? "Add as example" : "Add as example to correct", { icon: "plus", kind: "secondary", size: "sm", title: "Add this input as a training example, with this answer as its expected output", onclick: () => addAsExample(it, ok) }),
			),
		),
	);
}

export function tryPanel(run) {
	const box = h("section", { class: "card try-panel" });
	const render = () => {
		const pr = run.predict || {};
		const status = pr.status;
		const job = activeJob();
		const mine = Boolean(job && job.kind === "predict" && job.run_id === run.id);
		const text = h("textarea", {
			rows: vision(run) ? 2 : 4, value: S.tryInput,
			placeholder: vision(run) ? "Optional text to send with the image" : "Paste an input the model hasn't seen…",
			"aria-label": "Input to try",
		});
		text.addEventListener("input", () => (S.tryInput = text.value));
		mount(box,
			h("div", { class: "card-head" },
				h("div", {}, h("h3", { class: "section-title" }, "Try it"), h("p", { class: "muted small" }, "Run this model on your own input before exporting. Loading the model takes a moment.")),
			),
			h("div", { class: "card-body stack" },
				vision(run) ? h("div", { class: "try-inputs" }, imagePicker(render), text) : text,
				h("div", { class: "row" },
					button(mine ? "Running…" : "Run", { icon: mine ? null : "play", kind: "primary", disabled: Boolean(job) || Boolean(S.tryImage && S.tryImage.uploading), onclick: () => runTry(run) }),
					mine ? button("Stop", { icon: "stop", kind: "ghost", onclick: () => app.stopRun(run.id) }) : null,
					mine ? h("span", { class: "muted row-tight" }, h("span", { class: "spinner" }), (status && status.stage) || "Starting") : null,
					!mine && job ? h("span", { class: "faint" }, `Waiting for the ${JOB_LABEL[job.kind] || job.kind} job to finish`) : null,
				),
				!mine && status && status.state === "failed" ? callout("bad", "alert-circle", status.error || "The try failed. See the training log for details.") : null,
				!mine && pr.result ? pr.result.items.map((it) => result(run, it)) : null,
			),
		);
	};
	render();
	return box;
}
