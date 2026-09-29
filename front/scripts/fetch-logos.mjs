// Télécharge les logos des 30 équipes NBA depuis le CDN officiel vers
// public/logos/, pour les servir nous-mêmes : cdn.nba.com est bloqué sur
// certains réseaux (IUT, VPS OVH), le navigateur du visiteur ne doit donc
// pas dépendre de lui.
//
// Deux variantes par équipe : L (fond clair) et D (fond sombre, couleurs
// ajustées pour une dizaine d'équipes).
//
// Usage : npm run fetch:logos

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// Les ID NBA des 30 franchises se suivent (ATL = 1610612737 ... CHA = 1610612766).
const FIRST_TEAM_ID = 1610612737;
const LAST_TEAM_ID = 1610612766;
const VARIANTS = ["L", "D"];

const outDir = join(dirname(fileURLToPath(import.meta.url)), "..", "public", "logos");
await mkdir(outDir, { recursive: true });

let failures = 0;
for (let id = FIRST_TEAM_ID; id <= LAST_TEAM_ID; id++) {
  for (const variant of VARIANTS) {
    const url = `https://cdn.nba.com/logos/nba/${id}/global/${variant}/logo.svg`;
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(20_000) });
      const type = res.headers.get("content-type") ?? "";
      if (!res.ok || !type.includes("svg")) throw new Error(`HTTP ${res.status} (${type})`);
      await writeFile(join(outDir, `${id}-${variant}.svg`), await res.text());
    } catch (err) {
      failures++;
      console.error(`${id} ${variant} : échec — ${err.message}`);
    }
  }
}

const total = (LAST_TEAM_ID - FIRST_TEAM_ID + 1) * VARIANTS.length;
console.log(`${total - failures}/${total} logos enregistrés dans ${outDir}`);
if (failures) process.exit(1);
