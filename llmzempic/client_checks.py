"""Code the generated clients use to check data set fields against the value lists an export ships in
datasets/. Ollama can't enforce lists this long, so the clients do: an answer holding a value that isn't
on its list raises an error naming the field and the value, worded like the app's own checks."""

import json
import pprint

PYTHON = '''
# Data set fields and the list files (in the export's datasets/ folder) their values must come from.
REFS = __REFS__
EXPORT_DIR = Path(__file__).resolve().parent.parent
_LISTS = {}


def _found(value, path, where):
	if not path:
		yield where, value
	elif path[0] == "*":
		if isinstance(value, list):
			for i, item in enumerate(value):
				yield from _found(item, path[1:], where + [str(i)])
	elif isinstance(value, dict) and path[0] in value:
		yield from _found(value[path[0]], path[1:], where + [path[0]])


def check_values(result):
	"""Raise ValueError when a data set field holds a value that isn't on its list."""
	for ref in REFS:
		if ref["file"] not in _LISTS:
			_LISTS[ref["file"]] = set((EXPORT_DIR / ref["file"]).read_text(encoding="utf-8").splitlines())
		for where, value in _found(result, ref["path"], []):
			if isinstance(value, str) and value not in _LISTS[ref["file"]]:
				raise ValueError(f'{"/".join(where)}: {json.dumps(value, ensure_ascii=False)} isn\\'t in data set {ref["name"]}')
	return result
'''

PHP = """
// Data set fields and the list files (in the export's datasets/ folder) their values must come from.
const __CONST__ = <<<'JSON'
__REFS__
JSON;

// Throws when a data set field holds a value that isn't on its list.
function __FN___check_values(array $result): array
{
	static $lists = [];
	foreach (json_decode(__CONST__, true) as $ref) {
		$file = __DIR__ . '/../' . $ref['file'];
		$lists[$file] ??= array_flip(file($file, FILE_IGNORE_NEW_LINES));
		foreach (__FN___found($result, $ref['path'], []) as [$where, $value]) {
			if (is_string($value) && !isset($lists[$file][$value])) {
				throw new UnexpectedValueException(
					implode('/', $where) . ': ' . json_encode($value, JSON_UNESCAPED_UNICODE) . " isn't in data set {$ref['name']}"
				);
			}
		}
	}
	return $result;
}

function __FN___found(mixed $value, array $path, array $where): Generator
{
	if (!$path) {
		yield [$where, $value];
		return;
	}
	$step = array_shift($path);
	if ($step === '*') {
		if (is_array($value) && array_is_list($value)) {
			foreach ($value as $i => $item) {
				yield from __FN___found($item, $path, [...$where, (string) $i]);
			}
		}
	} elseif (is_array($value) && array_key_exists($step, $value)) {
		yield from __FN___found($value[$step], $path, [...$where, $step]);
	}
}
"""

CSHARP = """
	// Data set fields and the list files (in the export's datasets/ folder) their values must come from.
	private static readonly (string[] Fields, string List, string Name)[] Refs =
	{
__REFS__
	};
	private static readonly ConcurrentDictionary<string, HashSet<string>> Lists = new();

	/// <summary>Throws when a data set field holds a value that isn't on its list.</summary>
	private void CheckValues(string text)
	{
		using var document = JsonDocument.Parse(text);
		foreach (var (fields, list, name) in Refs)
		{
			var allowed = Lists.GetOrAdd(Path.Combine(_exportDir, list), f => new HashSet<string>(File.ReadLines(f), StringComparer.Ordinal));
			foreach (var (where, value) in Found(document.RootElement, fields, 0, ""))
			{
				if (value.ValueKind == JsonValueKind.String && !allowed.Contains(value.GetString()!))
				{
					throw new InvalidOperationException($"{where}: \\"{value.GetString()}\\" isn't in data set {name}");
				}
			}
		}
	}

	private static IEnumerable<(string Where, JsonElement Value)> Found(JsonElement value, string[] fields, int step, string where)
	{
		if (step == fields.Length)
		{
			yield return (where, value);
			yield break;
		}
		if (fields[step] == "*")
		{
			if (value.ValueKind != JsonValueKind.Array)
			{
				yield break;
			}
			var index = 0;
			foreach (var item in value.EnumerateArray())
			{
				foreach (var hit in Found(item, fields, step + 1, Join(where, index++.ToString(CultureInfo.InvariantCulture))))
				{
					yield return hit;
				}
			}
		}
		else if (value.ValueKind == JsonValueKind.Object && value.TryGetProperty(fields[step], out var child))
		{
			foreach (var hit in Found(child, fields, step + 1, Join(where, fields[step])))
			{
				yield return hit;
			}
		}
	}

	private static string Join(string where, string step) => where.Length == 0 ? step : where + "/" + step;
"""


def python_checks(refs):
	return PYTHON.replace("__REFS__", pprint.pformat(refs, indent=1, width=100, sort_dicts=False))


def php_checks(fn, refs):
	return PHP.replace("__CONST__", fn.upper() + "_REFS").replace("__FN__", fn).replace("__REFS__", json.dumps(refs, indent=4, ensure_ascii=False))


def csharp_checks(refs):
	# json.dumps output (ASCII with backslash escapes) is also a valid C# string literal.
	rows = [
		f"\t\t(new[] {{ {', '.join(json.dumps(p) for p in r['path'])} }}, {json.dumps(r['file'])}, {json.dumps(r['name'])}),"
		for r in refs
	]
	return CSHARP.replace("__REFS__", "\n".join(rows))
