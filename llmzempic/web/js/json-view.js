import { h } from "./dom.js";

// Pretty JSON as highlighted DOM (no innerHTML, so values are never interpreted as markup).
export function jsonNodes(value, indent = 0, out = []) {
	const pad = "  ".repeat(indent);
	const punct = (t) => out.push(h("span", { class: "j-punct" }, t));
	if (Array.isArray(value)) {
		if (!value.length) { punct("[]"); return out; }
		const inline = value.every((v) => v === null || typeof v !== "object") && JSON.stringify(value).length <= 60;
		punct("[");
		value.forEach((v, i) => {
			if (!inline) out.push(`\n${pad}  `);
			jsonNodes(v, indent + 1, out);
			if (i < value.length - 1) punct(inline ? ", " : ",");
		});
		if (!inline) out.push(`\n${pad}`);
		punct("]");
	} else if (value !== null && typeof value === "object") {
		const entries = Object.entries(value);
		if (!entries.length) { punct("{}"); return out; }
		punct("{");
		entries.forEach(([k, v], i) => {
			out.push(`\n${pad}  `);
			out.push(h("span", { class: "j-key" }, JSON.stringify(k)));
			punct(": ");
			jsonNodes(v, indent + 1, out);
			if (i < entries.length - 1) punct(",");
		});
		out.push(`\n${pad}`);
		punct("}");
	} else if (typeof value === "string") {
		out.push(h("span", { class: "j-str" }, JSON.stringify(value)));
	} else if (typeof value === "number") {
		out.push(h("span", { class: "j-num" }, String(value)));
	} else {
		out.push(h("span", { class: "j-lit" }, String(value)));
	}
	return out;
}

export const jsonBlock = (value, cls = "") => h("pre", { class: `code-block ${cls}`.trim() }, jsonNodes(value));
