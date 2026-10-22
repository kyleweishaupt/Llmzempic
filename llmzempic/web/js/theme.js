const KEY = "llmz-theme";
const media = matchMedia("(prefers-color-scheme: dark)");

export function currentTheme() {
	try { return localStorage.getItem(KEY) || "system"; } catch { return "system"; }
}

function resolve(theme) {
	return theme === "dark" || (theme === "system" && media.matches) ? "dark" : "light";
}

export function applyTheme(theme) {
	try {
		if (theme === "system") localStorage.removeItem(KEY);
		else localStorage.setItem(KEY, theme);
	} catch { /* private mode: the choice lasts for this page only */ }
	document.documentElement.dataset.theme = resolve(theme);
}

media.addEventListener("change", () => {
	if (currentTheme() === "system") document.documentElement.dataset.theme = resolve("system");
});
