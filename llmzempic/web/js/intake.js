import { uploadImage } from "./api.js";
import { $, button, callout, h, icon, mount } from "./dom.js";
import { toast } from "./feedback.js";
import { plural, randomId } from "./format.js";
import { outputForm } from "./output-form.js";
import { scheduleExamplesSave } from "./project.js";
import { blankOutput, labelFields } from "./schema-model.js";
import { S, isImageProject, schemaTree } from "./state.js";

const MAX_SIDE = 1280;
const KEEP_TYPES = ["image/jpeg", "image/png", "image/webp"];
const PARALLEL = 3;

// Orient (EXIF) and shrink big photos in the browser: uploads stay small and training sees what
// the user sees. Images that are already small and web-friendly are sent untouched.
export async function prepareImage(file) {
	let bitmap;
	try {
		bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
	} catch {
		throw new Error("isn't an image this browser can read");
	}
	const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
	if (scale === 1 && KEEP_TYPES.includes(file.type) && file.size <= 4 * 1024 * 1024) {
		bitmap.close();
		return file;
	}
	const canvas = document.createElement("canvas");
	canvas.width = Math.max(1, Math.round(bitmap.width * scale));
	canvas.height = Math.max(1, Math.round(bitmap.height * scale));
	const ctx = canvas.getContext("2d");
	ctx.fillStyle = "#fff"; // transparent PNGs become white, not black, as JPEG
	ctx.fillRect(0, 0, canvas.width, canvas.height);
	ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
	bitmap.close();
	return new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("couldn't be converted"))), "image/jpeg", 0.9));
}

export const imageFiles = (list) => [...(list || [])].filter((f) => f.type.startsWith("image/") || /\.(jpe?g|png|webp|gif|bmp|avif)$/i.test(f.name));

export async function uploadPrepared(file) {
	return uploadImage(S.pid, await prepareImage(file));
}

// Upload files with a few in flight; onDone(file, name) / onFail(file, error) fire per file.
async function uploadAll(files, { onDone, onFail, stopped }) {
	let next = 0;
	const worker = async () => {
		while (next < files.length && !stopped()) {
			const file = files[next++];
			try {
				onDone(file, await uploadPrepared(file));
			} catch (e) {
				onFail(file, e);
			}
		}
	};
	await Promise.all(Array.from({ length: PARALLEL }, worker));
}

export function openIntake(files) {
	files = imageFiles(files);
	if (!files.length) {
		toast("Those files aren't images.", true);
		return;
	}
	const parsed = schemaTree();
	const answer = structuredClone(S.lastLabel) || blankOutput(S.config.schema);
	let text = "";
	let stop = false;
	let running = false;

	const dlg = h("dialog", { class: "wide intake" });
	const urls = files.slice(0, 18).map((f) => URL.createObjectURL(f));
	const close = () => {
		if (running) return;
		urls.forEach(URL.revokeObjectURL);
		dlg.close();
		dlg.remove();
	};
	const textArea = h("textarea", { rows: 2, placeholder: "Usually empty. Use it when the instructions need extra context per image." });
	textArea.addEventListener("input", () => (text = textArea.value));
	const progressSlot = h("div", { class: "intake-progress", hidden: true });
	const addBtn = button(`Add ${plural(files.length, "example")}`, { kind: "primary", icon: "plus" });
	// Unlabeled imports would all need fixing one by one, so wait until each required label is set.
	const required = parsed.ok ? labelFields(parsed.root).filter((f) => f.required) : [];
	const labeled = () => required.every((f) => answer && answer[f.name] !== undefined);
	const syncAdd = () => { addBtn.disabled = running || !labeled(); };
	syncAdd();
	const cancelBtn = button("Cancel", { kind: "ghost", onclick: () => { if (running) stop = true; else close(); } });

	addBtn.addEventListener("click", async () => {
		running = true;
		addBtn.disabled = true;
		cancelBtn.textContent = "Stop";
		progressSlot.hidden = false;
		S.lastLabel = structuredClone(answer);
		const known = new Set(S.examples.map((e) => e.image).filter(Boolean));
		let done = 0, added = 0, skipped = 0;
		const failures = [];
		const bar = h("span", { style: { width: "0%" } });
		const label = h("span", { class: "muted small" });
		mount(progressSlot, h("div", { class: "progress" }, bar), label);
		const tick = () => {
			bar.style.width = `${(100 * done) / files.length}%`;
			label.textContent = `Uploaded ${done} of ${files.length}` + (failures.length ? `, ${failures.length} failed` : "");
		};
		tick();
		await uploadAll(files, {
			stopped: () => stop,
			onDone: (file, name) => {
				done++;
				if (known.has(name)) skipped++;
				else {
					known.add(name);
					S.examples.unshift({ id: randomId(), input: text, image: name, output: structuredClone(answer) });
					added++;
				}
				tick();
			},
			onFail: (file, e) => {
				done++;
				failures.push(`${file.name}: ${e.message}`);
				tick();
			},
		});
		running = false;
		S.page = 0;
		if (added) scheduleExamplesSave();
		document.dispatchEvent(new CustomEvent("examples-changed"));
		const summary = [`Added ${plural(added, "example")}.`];
		if (skipped) summary.push(`${plural(skipped, "image")} already in this project were skipped.`);
		if (stop) summary.push("Stopped before the rest were uploaded.");
		if (!failures.length) {
			close();
			toast(summary.join(" "));
			return;
		}
		cancelBtn.textContent = "Close";
		mount(progressSlot, callout("warn", "alert", h("b", {}, summary.join(" ")), ` ${plural(failures.length, "file")} couldn't be added:`,
			h("ul", {}, failures.slice(0, 12).map((f) => h("li", {}, f))), failures.length > 12 ? `…and ${failures.length - 12} more.` : null));
	});

	mount(dlg,
		h("div", { class: "dialog-head" },
			h("div", {}, h("h2", {}, `Add ${plural(files.length, "image")}`), h("p", {}, "Each image becomes an example with the answer you set here. You can change any of them later.")),
			button(null, { icon: "x", kind: "ghost", title: "Close", onclick: () => cancelBtn.click() }),
		),
		h("div", { class: "dialog-body" },
			h("div", { class: "intake-thumbs" }, urls.map((u) => h("img", { src: u, alt: "" })),
				files.length > urls.length ? h("span", { class: "more" }, `+${files.length - urls.length}`) : null),
			h("div", { class: "field" }, h("span", { class: "field-label" }, "Answer for all of them"),
				parsed.ok ? outputForm(parsed.root, answer, syncAdd, { keyPrefix: "intake" }) : h("p", { class: "faint" }, "Set answers per image after adding them.")),
			h("details", { class: "intake-text" }, h("summary", {}, "Same text for every image (optional)"), textArea),
			progressSlot,
		),
		h("div", { class: "dialog-foot" }, cancelBtn, addBtn),
	);
	dlg.addEventListener("cancel", (e) => { e.preventDefault(); cancelBtn.click(); });
	document.body.append(dlg);
	dlg.showModal();
}

export function pickImages(onFiles = openIntake) {
	const input = h("input", { type: "file", accept: "image/*", multiple: true });
	input.addEventListener("change", () => onFiles([...input.files]));
	input.click();
}

// Drag images anywhere onto the Examples tab, or paste them, to add them.
let overlay = null;
let depth = 0;
const active = () => S.pid && S.tab === "examples" && isImageProject() && !$("dialog[open]");
const hasFiles = (e) => [...(e.dataTransfer ? e.dataTransfer.types : [])].includes("Files");

document.addEventListener("dragenter", (e) => {
	if (!active() || !hasFiles(e)) return;
	depth++;
	if (!overlay) {
		overlay = h("div", { class: "drop-overlay" }, h("div", {}, icon("upload"), "Drop images to add them as examples"));
		document.body.append(overlay);
	}
});
// Files dropped outside a drop zone would otherwise make the browser navigate away to them.
document.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
document.addEventListener("dragleave", () => {
	if (!overlay) return;
	if (--depth <= 0) { overlay.remove(); overlay = null; depth = 0; }
});
document.addEventListener("drop", (e) => {
	if (overlay) { overlay.remove(); overlay = null; depth = 0; }
	if (!hasFiles(e)) return;
	e.preventDefault();
	if (active()) openIntake([...e.dataTransfer.files]);
});
document.addEventListener("paste", (e) => {
	if (!active() || e.target.closest("input, textarea")) return;
	const files = imageFiles([...e.clipboardData.files]);
	if (!files.length) return;
	e.preventDefault();
	openIntake(files);
});
