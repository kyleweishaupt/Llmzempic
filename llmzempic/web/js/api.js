async function parse(res) {
	const data = await res.json().catch(() => ({}));
	if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `${res.status} ${res.statusText}`);
	return data;
}

export async function api(path, { method = "GET", body } = {}) {
	const opts = { method, headers: {} };
	if (body !== undefined) {
		opts.headers["Content-Type"] = "application/json";
		opts.body = JSON.stringify(body);
	}
	return parse(await fetch("/api" + path, opts));
}

export async function uploadImage(pid, blob) {
	const res = await fetch(`/api/projects/${pid}/images`, {
		method: "POST",
		headers: { "Content-Type": blob.type || "application/octet-stream" },
		body: blob,
	});
	return (await parse(res)).name;
}

export const imageUrl = (pid, name) => `/api/projects/${encodeURIComponent(pid)}/images/${encodeURIComponent(name)}`;
export const runImageUrl = (pid, runId, name) => `/api/projects/${encodeURIComponent(pid)}/runs/${encodeURIComponent(runId)}/images/${encodeURIComponent(name)}`;
