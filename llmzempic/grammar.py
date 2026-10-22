"""GBNF grammar (llama.cpp) for a project's answers. Data set fields become prefix trees over their
values, so llama-server can hold 70k-1M allowed values in one grammar loaded at startup
(`--grammar-file`): a per-request enum that size never finishes compiling.

Keywords the grammar can't express (pattern, minLength, uniqueItems, ...) only make it more
permissive; clients still validate the answer against the schema."""

import itertools
import json
import os
import re

from .refs import REF_KEY

# llama.cpp's own primitive rules (examples/json_schema_to_grammar.py), so answers look the same.
PRIMITIVES = r"""space ::= | " " | "\n"{1,2} [ \t]{0,20}
boolean ::= ("true" | "false") space
null ::= "null" space
integral-part ::= [0] | [1-9] [0-9]{0,15}
decimal-part ::= [0-9]{1,16}
number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
integer ::= ("-"? integral-part) space
char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
string ::= "\"" char* "\"" space
value ::= object | array | string | number | boolean | null
object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? "}" space
array ::= "[" space ( value ("," space value)* )? "]" space"""

SIMPLE = {"string", "number", "integer", "boolean", "null"}


def literal(text):
	"""A GBNF string literal. GBNF knows \\" \\\\ \\n \\r \\t \\xHH but not JSON's \\b or \\f."""
	out = []
	for ch in text:
		if ch in '"\\':
			out.append("\\" + ch)
		elif ch == "\n":
			out.append("\\n")
		elif ch == "\r":
			out.append("\\r")
		elif ch == "\t":
			out.append("\\t")
		elif ord(ch) < 0x20 or ch == "\x7f":
			out.append(f"\\x{ord(ch):02X}")
		else:
			out.append(ch)
	return '"' + "".join(out) + '"'


def json_literal(value):
	return literal(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def value_rules(prefix, keys):
	"""Rules matching the rest of one JSON string, from after its opening quote through its closing quote,
	as a prefix tree with one rule per branching point. `keys` is the sorted, non-empty list of the
	strings' JSON-escaped contents. Returns (start rule, [(name, body)])."""
	names = (f"{prefix}-{n}" for n in itertools.count())
	start = next(names)
	rules = []
	pending = [(start, 0, len(keys), 0)]  # (rule, first key, end key, characters already matched)
	while pending:
		name, lo, hi, depth = pending.pop()
		alts = []
		if len(keys[lo]) == depth:  # the shared prefix is itself a value; it sorts first
			alts.append(literal('"'))
			lo += 1
		while lo < hi:
			ch = keys[lo][depth]
			end = lo + 1
			while end < hi and keys[end][depth] == ch:
				end += 1
			if end - lo == 1:
				alts.append(literal(keys[lo][depth:] + '"'))
			else:
				shared = os.path.commonprefix([keys[lo], keys[end - 1]])
				child = next(names)
				alts.append(f"{literal(shared[depth:])} {child}")
				pending.append((child, lo, end, len(shared)))
			lo = end
		rules.append((name, " | ".join(alts)))
	return start, rules


def _types(schema):
	declared = schema.get("type")
	if isinstance(declared, str):
		return [declared]
	return [t for t in declared if isinstance(t, str)] if isinstance(declared, list) else []


class _Builder:
	def __init__(self, values_for):
		self.values_for = values_for
		self.rules = {}
		self.sets = {}

	def add(self, name, body):
		name = re.sub(r"[^a-zA-Z0-9-]+", "-", name).strip("-") or "r"
		unique, n = name, 2
		while unique in self.rules and self.rules[unique] != body:
			unique, n = f"{name}{n}", n + 1
		self.rules[unique] = body
		return unique

	def data_set(self, dataset_id):
		if dataset_id not in self.sets:
			# Matched as the JSON text training wrote (json.dumps, ensure_ascii=False).
			keys = sorted({json.dumps(str(v), ensure_ascii=False)[1:-1] for v in self.values_for(dataset_id)})
			if not keys:
				raise ValueError(f'Data set "{dataset_id}" has no values')
			prefix = f"ds{len(self.sets)}"
			start, rules = value_rules(prefix, keys)
			self.rules.update(rules)
			self.sets[dataset_id] = self.add(prefix, f'"\\"" {start} space')
		return self.sets[dataset_id]

	def node(self, schema, name):
		"""The rule (or primitive) for one schema node."""
		if not isinstance(schema, dict):
			return "value"
		if isinstance(schema.get(REF_KEY), str):
			alts = [self.data_set(schema[REF_KEY])] + (["null"] if "null" in _types(schema) else [])
			return self.add(name, " | ".join(alts))
		if "const" in schema:
			return self.add(name, f"{json_literal(schema['const'])} space")
		if isinstance(schema.get("enum"), list) and schema["enum"]:
			return self.add(name, "(" + " | ".join(json_literal(v) for v in schema["enum"]) + ") space")
		for key in ("anyOf", "oneOf"):
			if isinstance(schema.get(key), list) and schema[key]:
				return self.add(name, " | ".join(self.node(s, f"{name}-{i}") for i, s in enumerate(schema[key])))
		types = _types(schema) or (["object"] if "properties" in schema else ["array"] if "items" in schema else [])
		if not types:
			return "value"
		alts = [self.typed(schema, t, name if len(types) == 1 else f"{name}-{t}") for t in types]
		return alts[0] if len(alts) == 1 else self.add(name, " | ".join(alts))

	def typed(self, schema, kind, name):
		if kind in SIMPLE:
			return kind
		if kind == "array":
			return self.array(schema, name)
		if kind == "object":
			return self.object(schema, name)
		return "value"

	def array(self, schema, name):
		item = self.node(schema.get("items"), f"{name}-item") if isinstance(schema.get("items"), dict) else "value"
		lo = max(0, int(schema.get("minItems") or 0))
		hi = schema.get("maxItems")
		hi = max(lo, int(hi)) if hi is not None else None
		more = f'("," space {item})'
		if hi == 0:
			return self.add(name, '"[" space "]" space')
		if lo == 0:
			inner = item if hi == 1 else f"{item} {more}" + ("*" if hi is None else f"{{0,{hi - 1}}}")
			return self.add(name, f'"[" space ({inner})? "]" space')
		inner = item if hi == 1 else f"{item} {more}" + (f"{{{lo - 1},}}" if hi is None else f"{{{lo - 1},{hi - 1}}}")
		return self.add(name, f'"[" space {inner} "]" space')

	def object(self, schema, name):
		props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
		extra = schema.get("additionalProperties", True) is not False
		if not props:
			return "object" if extra else self.add(name, '"{" space "}" space')
		required = set(schema.get("required") or [])
		keys = list(props)
		kv = [self.add(f"{name}-{k}-kv", f'{json_literal(k)} space ":" space {self.node(props[k], f"{name}-{k}")}') for k in keys]
		any_kv = 'string ":" space value'
		if not extra and required >= set(keys):
			return self.add(name, '"{" space ' + ' "," space '.join(kv) + ' "}" space')

		# Optional members keep schema order (as training wrote them); a comma goes before every member
		# but the first one written, so each position knows whether anything came before it.
		tails = {}

		def tail(i, first):
			if (i, first) in tails:
				return tails[(i, first)]
			if i == len(keys):
				more = f'("," space {any_kv})*' if extra else ""
				body = f"({any_kv} {more})?" if extra and first else more
			else:
				rest = tail(i + 1, False)
				take = ("" if first else '"," space ') + kv[i] + (f" {rest}" if rest else "")
				if keys[i] in required:
					body = take
				else:
					skip = tail(i + 1, first)
					body = f"({take} | {skip})" if skip else f"({take})?"
			tails[(i, first)] = self.add(f"{name}-tail{i}{'-first' if first else ''}", body) if body else ""
			return tails[(i, first)]

		return self.add(name, f'"{{" space {tail(0, True)} "}}" space')


def values_grammar(schema, values_for):
	"""GBNF for answers to `schema`. Fields marked with refs.REF_KEY only accept values_for(dataset id)."""
	builder = _Builder(values_for)
	top = builder.node(schema, "root")
	if top != "root":
		builder.rules["root"] = top
	ordered = {"root": builder.rules.pop("root"), **builder.rules}
	return "\n".join(f"{name} ::= {body}" for name, body in ordered.items()) + "\n" + PRIMITIVES + "\n"
