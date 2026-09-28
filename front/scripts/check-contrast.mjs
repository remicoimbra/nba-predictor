// Vérifie les contrastes WCAG de la palette définie dans src/app/globals.css,
// pour les deux thèmes (chaque variable est déclarée en light-dark(clair, sombre)).
//
// - Texte : >= 7:1 (AAA) sur chaque fond où il peut apparaître.
// - Marques graphiques, bordures de contrôles, anneau de focus : >= 3:1
//   (WCAG 1.4.11, contraste non textuel).
//
// Usage : npm run check:contrast   (code de sortie 1 si une paire échoue)

import { readFileSync } from "node:fs";

const css = readFileSync(new URL("../src/app/globals.css", import.meta.url), "utf8");

const tokens = {};
for (const [, name, light, dark] of css.matchAll(/--([\w-]+):\s*light-dark\((#[0-9a-f]{6}),\s*(#[0-9a-f]{6})\)/gi)) {
  tokens[name] = { light, dark };
}

function luminance(hex) {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function ratio(a, b) {
  const [l1, l2] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (l1 + 0.05) / (l2 + 0.05);
}

const BACKGROUNDS = ["bg", "surface", "surface-2"];
const checks = [
  // [premier plan, fonds, seuil]
  ...["text", "text-2", "accent", "danger-text"].map((fg) => [fg, BACKGROUNDS, 7]),
  ["on-accent", ["accent"], 7],
  ["text", ["home-wash", "away-wash"], 7],
  ...["home", "away", "line-strong", "focus"].map((fg) => [fg, BACKGROUNDS, 3]),
];

let failures = 0;
for (const theme of ["light", "dark"]) {
  console.log(`\n=== Thème ${theme === "light" ? "clair" : "sombre"} ===`);
  for (const [fg, bgs, min] of checks) {
    for (const bg of bgs) {
      if (!tokens[fg] || !tokens[bg]) {
        console.log(`  MANQUANT  --${fg} / --${bg}`);
        failures++;
        continue;
      }
      const r = ratio(tokens[fg][theme], tokens[bg][theme]);
      const ok = r >= min;
      if (!ok) failures++;
      console.log(
        `  ${ok ? "OK  " : "ÉCHEC"}  --${fg.padEnd(12)} sur --${bg.padEnd(10)} ${r.toFixed(2).padStart(5)}:1  (min ${min}:1)`,
      );
    }
  }
}

console.log(failures ? `\n${failures} paire(s) en échec.` : "\nToutes les paires passent.");
process.exit(failures ? 1 : 0);
