import { $, button, h, icon, mount } from "./dom.js";

// `action` ({ label, onclick }) adds a button, e.g. Undo; such toasts stay up longer.
export function toast(message, error = false, action = null) {
	let stack = $(".toasts");
	if (!stack) {
		stack = h("div", { class: "toasts", role: "status", "aria-live": "polite" });
		document.body.append(stack);
	}
	const dismiss = () => {
		el.classList.add("leaving");
		setTimeout(() => el.remove(), 220);
	};
	const el = h("div", { class: error ? "toast error" : "toast" },
		icon(error ? "alert-circle" : "check-circle"),
		h("span", { class: "grow" }, message),
		action ? h("button", { type: "button", class: "toast-action", onclick: () => { dismiss(); action.onclick(); } }, action.label) : null,
	);
	stack.append(el);
	setTimeout(dismiss, error ? 7000 : action ? 8000 : 2800);
}

export const toastError = (e) => toast(e instanceof Error ? e.message : String(e), true);

// Promise-based replacement for window.confirm with a styled, labelled action.
export function confirmAction({ title, text, action, danger = false }) {
	return new Promise((resolve) => {
		const dlg = h("dialog", { class: "confirm" });
		const done = (ok) => { dlg.close(); dlg.remove(); resolve(ok); };
		mount(dlg,
			h("div", { class: "dialog-head" }, h("div", {}, h("h2", {}, title), text ? h("p", {}, text) : null)),
			h("div", { class: "dialog-foot" },
				button("Cancel", { kind: "ghost", onclick: () => done(false) }),
				button(action, { kind: danger ? "danger" : "primary", onclick: () => done(true) }),
			),
		);
		dlg.addEventListener("cancel", (e) => { e.preventDefault(); done(false); });
		document.body.append(dlg);
		dlg.showModal();
		dlg.querySelector(".dialog-foot .btn:last-child").focus();
	});
}

// Debounced saves keyed by kind; flushed before switching project or starting a job.
export const saver = {
	timers: {},
	listeners: new Set(),
	pending() {
		return Object.keys(this.timers).length > 0;
	},
	notify() {
		for (const fn of this.listeners) fn(this.pending());
	},
	schedule(key, fn, ms = 600) {
		if (this.timers[key]) clearTimeout(this.timers[key].t);
		const entry = { fn };
		entry.t = setTimeout(() => {
			delete this.timers[key];
			Promise.resolve(fn()).finally(() => this.notify());
		}, ms);
		this.timers[key] = entry;
		this.notify();
	},
	async flush() {
		const entries = Object.values(this.timers);
		this.timers = {};
		for (const e of entries) {
			clearTimeout(e.t);
			await e.fn();
		}
		this.notify();
	},
};
