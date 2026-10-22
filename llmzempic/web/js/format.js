export const pct = (v) => (v === null || v === undefined || Number.isNaN(v) ? "–" : `${Math.round(v * 100)}%`);

export const clip = (text, n) => {
	text = String(text ?? "");
	return text.length > n ? text.slice(0, n - 1) + "…" : text;
};

export const plural = (n, one, many = one + "s") => `${n.toLocaleString()} ${n === 1 ? one : many}`;

export const labelText = (v) => (v === true ? "Yes" : v === false ? "No" : v === null ? "Empty" : typeof v === "string" ? v : JSON.stringify(v));

export function bytes(n) {
	const units = ["B", "KB", "MB", "GB"];
	let i = 0;
	while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
	return `${n.toFixed(i ? 1 : 0)} ${units[i]}`;
}

export function duration(sec) {
	if (sec === null || sec === undefined) return "";
	sec = Math.max(0, Math.round(sec));
	if (sec < 60) return `${sec}s`;
	if (sec < 3600) return `${Math.floor(sec / 60)}m ${sec % 60}s`;
	return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
}

const dateFmt = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });

export const when = (ts) => (ts ? dateFmt.format(new Date(ts * 1000)) : "");

export function ago(ts) {
	if (!ts) return "";
	const sec = Date.now() / 1000 - ts;
	if (sec < 60) return "just now";
	if (sec < 3600) return `${Math.floor(sec / 60)} min ago`;
	if (sec < 86400) return `${Math.floor(sec / 3600)} h ago`;
	return when(ts);
}

export const randomId = () => Math.random().toString(16).slice(2, 12);

// "Qwen/Qwen2.5-1.5B-Instruct" -> "Qwen2.5-1.5B-Instruct"
export const shortModel = (id) => String(id || "").split(/[\\/]/).pop();
