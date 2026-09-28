"use client";

import { motion } from "motion/react";
import DataTable from "@/components/charts/DataTable";
import { useChartTransition } from "@/lib/charts";
import { int, num, pct, signed } from "@/lib/format";

// Statistiques comparées. `domain` : bornes de l'échelle de la barre
// (valeurs hors bornes écrêtées) ; `lowerIsBetter` pour les points encaissés.
const METRICS = [
  { key: "elo", label: "Rating Elo", domain: [1300, 1800], format: int },
  { key: "win_pct", label: "% victoires (saison)", domain: [0, 1], format: pct },
  {
    key: "win_pct_context",
    label: "% victoires à domicile / à l'extérieur",
    domain: [0, 1],
    format: pct,
  },
  { key: "win_pct_last10", label: "% victoires (10 derniers)", domain: [0, 1], format: pct },
  { key: "avg_pts_scored_last10", label: "Points marqués (moy. 10 derniers)", domain: [95, 130], format: num },
  {
    key: "avg_pts_allowed_last10",
    label: "Points encaissés (moy. 10 derniers)",
    hint: "moins = mieux",
    domain: [95, 130],
    format: num,
    lowerIsBetter: true,
  },
  { key: "net_rtg_last10", label: "Net Rating (10 derniers)", domain: [-15, 15], format: (v) => signed(v) },
  {
    key: "rest_days",
    label: "Jours de repos",
    domain: [0, 5],
    format: int,
    // Date passée interrogée avec l'état actuel des équipes : repos négatif, sans objet.
    valid: (v) => v >= 0,
  },
];

function share(value, [min, max]) {
  return Math.min(Math.max((value - min) / (max - min), 0.02), 1);
}

export default function ComparisonBars({ home, away }) {
  const transition = useChartTransition();
  const rows = METRICS.map((m) => {
    const valid = (v) => v != null && (!m.valid || m.valid(v));
    const h = valid(home.stats[m.key]) ? home.stats[m.key] : null;
    const a = valid(away.stats[m.key]) ? away.stats[m.key] : null;
    let better = null;
    if (h != null && a != null && h !== a) {
      better = (h > a) !== Boolean(m.lowerIsBetter) ? "home" : "away";
    }
    return { ...m, h, a, better };
  });

  return (
    <div>
      <div aria-hidden="true" className="mb-4 flex justify-between text-sm font-semibold">
        <span className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-sm bg-home" /> {home.tricode} · domicile
        </span>
        <span className="flex items-center gap-2">
          extérieur · {away.tricode} <span className="h-3 w-3 rounded-sm bg-away" />
        </span>
      </div>

      <ul className="space-y-4">
        {rows.map((r, i) => (
          <li key={r.key}>
            <p className="mb-1 text-center text-sm font-semibold">
              {r.label}
              {r.hint && <span className="font-normal text-text-2"> ({r.hint})</span>}
            </p>
            <div className="grid grid-cols-[3.5rem_1fr_1fr_3.5rem] items-center gap-2">
              <Value value={r.h} format={r.format} better={r.better === "home"} align="left" team={home} />
              <div aria-hidden="true" className="flex h-3 justify-end rounded-l-full bg-surface-2">
                {r.h != null && (
                  <motion.span
                    className="rounded-l-full bg-home"
                    initial={{ width: 0 }}
                    whileInView={{ width: `${share(r.h, r.domain) * 100}%` }}
                    viewport={{ once: true, margin: "-40px" }}
                    transition={transition(i * 0.05)}
                  />
                )}
              </div>
              <div aria-hidden="true" className="flex h-3 rounded-r-full bg-surface-2">
                {r.a != null && (
                  <motion.span
                    className="rounded-r-full bg-away"
                    initial={{ width: 0 }}
                    whileInView={{ width: `${share(r.a, r.domain) * 100}%` }}
                    viewport={{ once: true, margin: "-40px" }}
                    transition={transition(i * 0.05)}
                  />
                )}
              </div>
              <Value value={r.a} format={r.format} better={r.better === "away"} align="right" team={away} />
            </div>
          </li>
        ))}
      </ul>

      <DataTable
        caption="Statistiques comparées des deux équipes"
        columns={["Statistique", home.tricode, away.tricode, "Avantage"]}
        rows={rows.map((r) => [
          r.hint ? `${r.label} (${r.hint})` : r.label,
          r.h == null ? "n/d" : r.format(r.h),
          r.a == null ? "n/d" : r.format(r.a),
          r.better ? (r.better === "home" ? home.tricode : away.tricode) : "—",
        ])}
      />
    </div>
  );
}

// Valeur chiffrée ; l'avantage est marqué en gras + triangle (pas seulement
// par la couleur).
function Value({ value, format, better, align, team }) {
  return (
    <span className={`text-sm tabular-nums ${align === "right" ? "text-right" : ""} ${better ? "font-bold" : "text-text-2"}`}>
      {align === "right" && better && <Triangle />}
      {value == null ? "n/d" : format(value)}
      {align === "left" && better && <Triangle />}
      {better && <span className="sr-only"> (avantage {team.tricode})</span>}
    </span>
  );
}

function Triangle() {
  return (
    <svg viewBox="0 0 10 10" className="mx-1 inline h-2 w-2 align-middle" aria-hidden="true">
      <path d="M5 1l4 8H1z" fill="currentColor" />
    </svg>
  );
}
