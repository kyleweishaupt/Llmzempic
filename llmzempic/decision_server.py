"""`llmzempic serve`: a Decisions API over a decision run (base model + LoRA adapter) or over any base
model read training-free. The model stays loaded, so each request is one batched forward pass; requests
are answered one at a time.

POST /v1/decisions  {"input": ..., "questions": [...], "min_confidence": 0.7}  ->  {"answers": [...], "usage": {...}}
GET  /v1/models     the served model and its trained questions
"""

import threading

from . import decisions

DEFAULT_PORT = 7871


def create_decision_app(reader, name, trained=None, temperature=1.0):
	from fastapi import Body, FastAPI
	from fastapi.exceptions import RequestValidationError
	from fastapi.responses import JSONResponse

	app = FastAPI(title="Llmzempic decisions")
	lock = threading.Lock()
	trained = trained or []

	def error(status, message):
		return JSONResponse({"error": {"message": message, "type": "invalid_request_error"}}, status_code=status)

	@app.exception_handler(RequestValidationError)
	async def invalid_body(request, exc):
		return error(400, "The request body must be a JSON object.")

	@app.get("/health")
	def health():
		return {"status": "ok", "model": name}

	@app.get("/v1/models")
	def models():
		return {"object": "list", "data": [{
			"id": name, "object": "model", "owned_by": "llmzempic",
			"questions": [decisions.api_question(q) for q in trained],
		}]}

	@app.post("/v1/decisions")
	def decide(body: dict = Body(...)):
		try:
			state, questions, threshold = decisions.parse_request(body, trained)
		except decisions.DecisionError as e:
			return error(400, str(e))
		with lock:
			scores, tokens = reader.score(state, questions)
		return decisions.build_response(name, questions, scores, tokens, temperature, threshold)

	return app
