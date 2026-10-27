"""`llmzempic serve`: a Decisions API over decision runs or exports (base model + LoRA adapter), or any base
model read training-free. Models stay loaded, so each request is one batched forward pass; requests are
answered one at a time. Several models can be served at once: runs on the same base model share its
weights and switch LoRA adapters per request, an A/B route splits traffic between models, and an ensemble
averages their probabilities.

POST /v1/decisions  {"model": ..., "input": ..., "questions": [...], "min_confidence": 0.7 | "auto"}
POST /v1/feedback   {"id": "dec_...", "answers": {"intent": "billing"}}   the right answers, for monitoring
GET  /v1/models     served models, routes and their trained questions
GET  /v1/stats      ?model=: answer distribution, drift, confidence, refusals, accuracy from feedback
"""

import hashlib
import math
import random
import threading
import time
from collections import OrderedDict

from . import decisions
from .decision_monitor import monitor_stats

DEFAULT_PORT = 7871
REMEMBERED_IDS = 100_000


def same_question(a, b):
	return a["name"] == b["name"] and a["type"] == b["type"] and a["options"] == b["options"]


class Ensemble:
	"""Averages the calibrated probabilities of several models. Its trained questions are those every member
	was trained on with the same options."""

	def __init__(self, name, members):
		self.name = name
		self.members = members
		self.questions = [q for q in members[0].questions if all(any(same_question(q, o) for o in m.questions) for m in members[1:])]
		self.curves = {}
		self.labels = members[0].labels if hasattr(members[0], "labels") else {}
		if len({m.vision for m in members}) > 1:
			raise decisions.DecisionError(f'Ensemble "{name}" mixes image and text models.')

	@property
	def vision(self):
		return self.members[0].vision

	def answer(self, state, questions, settings=None, image=None):
		settings = settings or decisions.parse_settings({})
		total = [[0.0] * len(q["options"]) for q in questions]
		usage = {"input_tokens": 0, "cached_tokens": 0}
		for m in self.members:
			scores, used = m.read(state, questions, settings, image)
			usage["input_tokens"] += used["input_tokens"]
			usage["cached_tokens"] += used.get("cached_tokens", 0)
			for qi, (q, s) in enumerate(zip(questions, scores)):
				for oi, p in enumerate(decisions.softmax(s, decisions.question_temperature(q, m.temperature))):
					total[qi][oi] += p / len(self.members)
		logs = [[math.log(max(p, 1e-30)) for p in probs] for probs in total]
		return decisions.build_response(self.name, questions, logs, usage, 1.0, settings)


class Route:
	"""An A/B split: each request goes to one model, picked by weight (by a hash of "user" when the request
	names one, so a user keeps getting the same model)."""

	def __init__(self, name, members, weights):
		self.name = name
		self.members = members
		self.weights = weights

	def pick(self, user=None):
		if user is None:
			return random.choices(self.members, self.weights)[0]
		spot = int(hashlib.sha256(f"{self.name}:{user}".encode()).hexdigest(), 16) % 10_000 / 10_000 * sum(self.weights)
		for member, weight in zip(self.members, self.weights):
			spot -= weight
			if spot < 0:
				return member
		return self.members[-1]


class Router:
	def __init__(self, models, routes=(), default=None):
		self.models = {m.name: m for m in models}
		self.routes = {r.name: r for r in routes}
		self.default = default or (routes[0].name if routes else models[0].name)

	def resolve(self, name=None, user=None):
		"""(model that answers, route name or None)."""
		name = name or self.default
		if name in self.routes:
			return self.routes[name].pick(user), name
		if name in self.models:
			return self.models[name], None
		raise decisions.DecisionError(f'Unknown model "{name}". Served: {", ".join([*self.models, *self.routes])}.')


def create_decision_app(served, routes=(), logs=None, default=None):
	"""served: models (DecisionModel or Ensemble, or one of them); routes: A/B Routes; logs: {model name:
	DecisionLog} for the models whose decisions are logged."""
	from fastapi import Body, FastAPI
	from fastapi.exceptions import RequestValidationError
	from fastapi.responses import JSONResponse

	models = served if isinstance(served, (list, tuple)) else [served]
	router = Router(models, list(routes), default)
	logs = logs or {}
	app = FastAPI(title="Llmzempic decisions")
	lock = threading.Lock()
	recent = OrderedDict()  # decision id -> model name, so feedback finds its log

	def error(status, message):
		return JSONResponse({"error": {"message": message, "type": "invalid_request_error"}}, status_code=status)

	@app.exception_handler(RequestValidationError)
	async def invalid_body(request, exc):
		return error(400, "The request body must be a JSON object.")

	@app.get("/health")
	def health():
		return {"status": "ok", "models": list(router.models), "default": router.default}

	@app.get("/v1/models")
	def list_models():
		data = [{
			"id": m.name, "object": "model", "owned_by": "llmzempic", "input": "image" if m.vision else "text",
			"questions": [decisions.api_question(q) for q in m.questions],
			**({"members": [x.name for x in m.members]} if isinstance(m, Ensemble) else {}),
		} for m in router.models.values()]
		data += [{
			"id": r.name, "object": "route", "owned_by": "llmzempic",
			"variants": [{"model": m.name, "weight": w} for m, w in zip(r.members, r.weights)],
		} for r in router.routes.values()]
		return {"object": "list", "default": router.default, "data": data}

	@app.post("/v1/decisions")
	def decide(body: dict = Body(...)):
		try:
			model, route = router.resolve(body.get("model"), body.get("user"))
			state, questions, settings = decisions.parse_request(body, model.questions)
			if settings["image"] is not None and not model.vision:
				raise decisions.DecisionError(f'"{model.name}" reads text only; images in the input aren\'t supported.')
			if settings["image"] is None and model.vision:
				raise decisions.DecisionError(f'"{model.name}" decides about an image: add one to the input.')
			with lock:
				started = time.perf_counter()
				response = model.answer(state, questions, settings)
				latency = (time.perf_counter() - started) * 1000
		except decisions.DecisionError as e:
			return error(400, str(e))
		if route:
			response["route"] = route
		log = logs.get(model.name)
		if log is not None:
			log.record(response, latency, route, state, settings["image"])
			recent[response["id"]] = model.name
			while len(recent) > REMEMBERED_IDS:
				recent.popitem(last=False)
		return response

	@app.post("/v1/feedback")
	def feedback(body: dict = Body(...)):
		decision_id, answers = body.get("id"), body.get("answers")
		if not isinstance(decision_id, str) or not isinstance(answers, dict) or not answers:
			return error(400, 'Feedback needs the decision "id" and "answers": {question: right answer}.')
		name = recent.get(decision_id) or body.get("model")
		if name not in logs:
			return error(404, 'Unknown decision id. Send "model" too if the server restarted since the decision.')
		logs[name].feedback(decision_id, answers)
		return {"ok": True}

	@app.get("/v1/stats")
	def stats(model: str = None):
		name = model or (router.default if router.default in logs else next(iter(logs), None))
		if name not in logs:
			return error(404, "Decisions of that model aren't logged.")
		served_model = router.models.get(name)
		return {"model": name, **monitor_stats(logs[name].folder, getattr(served_model, "labels", None))}

	return app


def parse_named(values):
	"""[(name or None, value)] from repeated "[NAME=]VALUE" options; a value may itself contain "=" only
	when a name is given."""
	out = []
	for text in values:
		name, sep, value = text.partition("=")
		if sep and name and "/" not in name and "\\" not in name:
			out.append((name.strip(), value.strip()))
		else:
			out.append((None, text.strip()))
	return out


def parse_route(text, ensemble=False):
	"""("name", [(member, weight)]) from "NAME=A:90,B:10" (A/B weights) or "NAME=A,B" (ensemble members)."""
	name, sep, members = text.partition("=")
	if not sep or not name.strip() or not members.strip():
		raise decisions.DecisionError(f'Write "{"NAME=A,B" if ensemble else "NAME=A:90,B:10"}", not "{text}".')
	out = []
	for item in members.split(","):
		member, colon, weight = item.strip().partition(":")
		if ensemble and colon:
			raise decisions.DecisionError(f'Ensemble members take no weights: "{text}".')
		try:
			value = float(weight) if colon else 1.0
		except ValueError:
			raise decisions.DecisionError(f'"{weight}" isn\'t a weight in "{text}".')
		if value <= 0:
			raise decisions.DecisionError(f'Weights must be above 0 in "{text}".')
		out.append((member.strip(), value))
	if len(out) < 2:
		raise decisions.DecisionError(f'"{name}" needs at least two models.')
	return name.strip(), out


def unique_names(models):
	"""Rename models that share a name: name, name-2, name-3…"""
	seen = {}
	for m in models:
		n = seen.get(m.name, 0) + 1
		seen[m.name] = n
		if n > 1:
			m.name = f"{m.name}-{n}"
	return models
