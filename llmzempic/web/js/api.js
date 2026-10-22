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

// Upload a file as the raw request body with progress (fetch can't report upload progress).
export function uploadFile(url, file, onProgress, method = "POST") {
	return new Promise((resolve, reject) => {
		const xhr = new XMLHttpRequest();
		xhr.open(method, "/api" + url);
		xhr.setRequestHeader("Content-Type", "application/octet-stream");
		xhr.upload.onprogress = (e) => { if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total); };
		xhr.onload = () => {
			let data = {};
			try { data = JSON.parse(xhr.responseText || "{}"); } catch { /* not JSON */ }
			if (xhr.status >= 200 && xhr.status < 300) resolve(data);
			else reject(new Error(typeof data.detail === "string" ? data.detail : `${xhr.status} ${xhr.statusText}`));
		};
		xhr.onerror = () => reject(new Error("The upload was interrupted."));
		xhr.send(file);
	});
}
