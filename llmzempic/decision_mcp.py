"""`llmzempic mcp`: a decision model as a Model Context Protocol tool, so agents (Claude Code, Claude Desktop,
any MCP client) can ask it typed questions over stdio. One tool, `decide`, takes the same input, questions and
settings as POST /v1/decisions and returns the same answers.
"""

import json
import sys

from . import __version__, decisions

PROTOCOL = "2025-06-18"


def tool_definition(served):
	trained = served.questions
	listed = "; ".join(
		f"{q['name']} ({q['type']}{': ' + ', '.join(map(str, q['options'])) if q['type'] != 'predicate' else ''})" for q in trained)
	description = (
		f"Ask the decision model {served.name} typed questions about an input. Every option gets a calibrated "
		"probability; nothing is generated, so it is fast and cheap. "
		+ (f"Trained questions: {listed}. Leave questions out to ask all of them. " if trained else "Bring your own questions. ")
		+ "min_confidence \"auto\" refuses answers below each question's threshold for 90% held-out accuracy; "
		"a refused answer means: decide another way."
	)
	question = {
		"type": "object",
		"properties": {
			"name": {"type": "string"},
			"type": {"type": "string", "enum": list(decisions.TYPES)},
			"instructions": {"type": "string"},
			"choices": {"type": "array", "items": {"type": "string"}},
			"levels": {"type": "array", "items": {"type": "string"}},
		},
		"required": ["name"],
	}
	return {
		"name": "decide",
		"title": f"Decide with {served.name}",
		"description": description,
		"inputSchema": {
			"type": "object",
			"properties": {
				"input": {"type": "string", "description": "The text (or JSON) to decide about"},
				"questions": {"type": "array", "items": question, "description": "Questions to answer (default: all trained ones)"},
				"min_confidence": {"description": 'A number from 0 to 1, or "auto"'},
				"samples": {"type": "integer", "minimum": 1, "maximum": decisions.MAX_SAMPLES},
			},
			"required": ["input"],
		},
	}


def result(message_id, value):
	return {"jsonrpc": "2.0", "id": message_id, "result": value}


def failure(message_id, code, text):
	return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": text}}


def handle(message, served):
	"""The response to one JSON-RPC message, or None for notifications."""
	if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
		return failure(None, -32600, "Invalid request")
	method, message_id = message.get("method"), message.get("id")
	if message_id is None:
		return None  # notifications (initialized, cancelled) need no answer
	params = message.get("params") or {}
	if method == "initialize":
		return result(message_id, {
			"protocolVersion": params.get("protocolVersion") or PROTOCOL,
			"capabilities": {"tools": {"listChanged": False}},
			"serverInfo": {"name": "llmzempic", "title": f"Llmzempic decisions: {served.name}", "version": __version__},
		})
	if method == "ping":
		return result(message_id, {})
	if method == "tools/list":
		return result(message_id, {"tools": [tool_definition(served)]})
	if method == "tools/call":
		if params.get("name") != "decide":
			return failure(message_id, -32602, f"Unknown tool: {params.get('name')}")
		try:
			state, questions, settings = decisions.parse_request(params.get("arguments") or {}, served.questions)
			response = served.answer(state, questions, settings)
		except decisions.DecisionError as e:
			return result(message_id, {"content": [{"type": "text", "text": str(e)}], "isError": True})
		return result(message_id, {
			"content": [{"type": "text", "text": json.dumps(response["answers"], ensure_ascii=False)}],
			"structuredContent": response,
			"isError": False,
		})
	return failure(message_id, -32601, f"Method not found: {method}")


def run_stdio(served, stdin=None, stdout=None):
	"""Newline-delimited JSON-RPC over stdio until stdin closes."""
	stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
	for line in stdin:
		if not line.strip():
			continue
		try:
			message = json.loads(line)
		except ValueError:
			reply = failure(None, -32700, "Parse error")
		else:
			reply = handle(message, served)
		if reply is not None:
			stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
			stdout.flush()
