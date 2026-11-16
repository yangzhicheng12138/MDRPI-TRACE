// Applied before first paint; themes and drafts stay in this browser.
try {
  const value = localStorage.getItem("mdrpi.appearance") || "system";
  document.documentElement.dataset.theme = value === "system" ? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light") : value;
} catch (_) { document.documentElement.dataset.theme = "light"; }
