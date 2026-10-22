"""Identifiers and C# result types shared by the generated clients."""

import json
import re

CS_SIMPLE = {"string": "string", "integer": "long", "number": "double", "boolean": "bool"}


def pascal(name):
	words = re.split(r"[^A-Za-z0-9]+", name)
	ident = "".join(w[:1].upper() + w[1:] for w in words if w)
	return ident if ident[:1].isalpha() else f"Model{ident}"


def snake(name):
	ident = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
	return ident if ident[:1].isalpha() else f"model_{ident}"


def _cs_type(schema, name_hint, classes):
	"""Map a JSON Schema node to a C# type, generating classes for nested objects."""
	types = schema.get("type")
	types = types if isinstance(types, list) else [types]
	nullable = "null" in types
	main = next((t for t in types if t and t != "null"), None)
	if main is None and "enum" in schema:
		main = "string"
	if main == "object" and isinstance(schema.get("properties"), dict):
		return _cs_class(schema, name_hint, classes) + ("?" if nullable else ""), "" if nullable else " = new();"
	if main == "array":
		item = _cs_type(schema.get("items") or {}, name_hint + "Item", classes)[0]
		return f"List<{item}>" + ("?" if nullable else ""), "" if nullable else " = new();"
	if main in CS_SIMPLE:
		cs = CS_SIMPLE[main]
		if nullable:
			return cs + "?", ""
		return cs, ' = "";' if cs == "string" else ""
	return "JsonElement", ""


def _cs_class(schema, name, classes):
	base, n = name, 2
	while name in classes:
		name, n = f"{base}{n}", n + 1
	classes[name] = ""
	lines = [f"public sealed class {name}", "{"]
	for key, prop in schema["properties"].items():
		prop = prop if isinstance(prop, dict) else {}
		member = pascal(key)
		if member == name:
			member += "Value"
		cs_type, init = _cs_type(prop, member, classes)
		if prop.get("enum"):
			values = ", ".join(json.dumps(v) for v in prop["enum"])
			lines.append(f"\t/// <summary>One of: {values}</summary>")
		lines.append(f'\t[JsonPropertyName("{key}")]')
		lines.append(f"\tpublic {cs_type} {member} {{ get; set; }}{init}")
	lines.append("}")
	classes[name] = "\n".join(lines)
	return name


def cs_result(schema, cls):
	"""(result type, class definitions, deserialize statement) for a client's typed result."""
	classes = {}
	result_type = "JsonElement"
	if schema.get("type") == "object" and isinstance(schema.get("properties"), dict):
		result_type = _cs_class(schema, f"{cls}Result", classes)
	deserialize = (
		"using var result = JsonDocument.Parse(text);\n\t\treturn result.RootElement.Clone();"
		if result_type == "JsonElement"
		else f"return JsonSerializer.Deserialize<{result_type}>(text)\n\t\t\t?? throw new InvalidOperationException(\"Model returned null.\");"
	)
	return result_type, "\n\n".join(classes.values()), deserialize
