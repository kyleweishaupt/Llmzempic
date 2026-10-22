import { h, mount, s } from "./dom.js";

function niceTicks(max, count) {
	const raw = max / count;
	const mag = Math.pow(10, Math.floor(Math.log10(raw)));
	const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((v) => v >= raw) || raw;
	const ticks = [];
	for (let v = 0; v < max + step - 1e-9; v += step) ticks.push(Number(v.toPrecision(6)));
	return ticks;
}

// Training loss per step (line) with held-out loss per epoch (dots), plus a hover readout.
export function lossChart(events) {
	const train = events.filter((e) => e.loss !== undefined).map((e) => [e.step, e.loss]);
	const evals = events.filter((e) => e.eval_loss !== undefined).map((e) => [e.step, e.eval_loss]);
	if (!train.length && !evals.length) return null;
	const W = 800, H = 240, m = { l: 44, r: 12, t: 12, b: 28 };
	const all = train.concat(evals);
	const maxX = Math.max(1, ...all.map((p) => p[0]));
	const ticksY = niceTicks(Math.max(...all.map((p) => p[1])) * 1.05 || 1, 4);
	const maxY = ticksY[ticksY.length - 1] || 1;
	const x = (v) => m.l + (v / maxX) * (W - m.l - m.r);
	const y = (v) => H - m.b - (v / maxY) * (H - m.t - m.b);
	const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Training loss by step", preserveAspectRatio: "none" });
	for (const t of ticksY) {
		svg.append(s("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: y(t), y2: y(t) }));
		svg.append(s("text", { class: "axis", x: m.l - 8, y: y(t) + 4, "text-anchor": "end" }, t));
	}
	const stepX = Math.max(1, Math.ceil(maxX / 8));
	for (let v = 0; v <= maxX; v += stepX) svg.append(s("text", { class: "axis", x: x(v), y: H - 8, "text-anchor": "middle" }, v));
	if (train.length) {
		const d = train.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
		const area = `${d}L${x(train[train.length - 1][0]).toFixed(1)},${y(0)}L${x(train[0][0]).toFixed(1)},${y(0)}Z`;
		svg.append(s("path", { d: area, class: "loss-area" }));
		svg.append(s("path", { d, class: "loss-line" }));
	}
	for (const p of evals) svg.append(s("circle", { cx: x(p[0]), cy: y(p[1]), r: 5, class: "eval-dot" }));
	const cross = s("line", { x1: 0, x2: 0, y1: m.t, y2: H - m.b, class: "crosshair", visibility: "hidden" });
	const dot = s("circle", { r: 5, class: "hover-dot", visibility: "hidden" });
	const hit = s("rect", { x: m.l, y: 0, width: W - m.l - m.r, height: H, fill: "transparent" });
	svg.append(cross, dot, hit);

	const tip = h("div", { class: "chart-tip", hidden: true });
	const legend = h("div", { class: "legend" },
		h("span", { class: "line" }, "Training loss"),
		evals.length ? h("span", { class: "dot" }, "Held-out loss, per epoch") : null,
	);
	const points = train.length ? train : evals;
	hit.addEventListener("mousemove", (e) => {
		const rect = svg.getBoundingClientRect();
		const vx = ((e.clientX - rect.left) / rect.width) * W;
		let best = points[0];
		for (const p of points) if (Math.abs(x(p[0]) - vx) < Math.abs(x(best[0]) - vx)) best = p;
		for (const [k, v] of [["x1", x(best[0])], ["x2", x(best[0])]]) cross.setAttribute(k, v);
		dot.setAttribute("cx", x(best[0]));
		dot.setAttribute("cy", y(best[1]));
		cross.setAttribute("visibility", "visible");
		dot.setAttribute("visibility", "visible");
		const evalAt = evals.find((p) => p[0] === best[0]);
		mount(tip, h("b", {}, `Step ${best[0]}`), h("br"), `${train.length ? "Training" : "Held-out"} loss ${best[1].toFixed(4)}`,
			evalAt && train.length ? h("span", {}, h("br"), `Held-out loss ${evalAt[1].toFixed(4)}`) : null);
		tip.hidden = false;
		const px = (x(best[0]) / W) * rect.width;
		tip.style.left = `${Math.min(px + 14, rect.width - 180)}px`;
	});
	hit.addEventListener("mouseleave", () => {
		tip.hidden = true;
		cross.setAttribute("visibility", "hidden");
		dot.setAttribute("visibility", "hidden");
	});
	return h("div", { class: "chart" }, legend, h("div", { class: "chart-plot" }, svg, tip));
}
