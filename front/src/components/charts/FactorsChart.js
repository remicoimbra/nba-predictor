"use client";

import { motion } from "motion/react";
import DataTable from "@/components/charts/DataTable";
import { useChartTransition } from "@/lib/charts";
import { pct } from "@/lib/format";

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

/**
 * Impact d'un facteur en points de probabilité : proba du modèle, moins la
 * proba qu'il donnerait sans ce facteur. Les contributions de l'API sont en
 * log-odds (valeurs de type SHAP d'XGBoost, additives avant la sigmoïde) :
 * cette conversion rend l'ordre de grandeur lisible (« +12 pts »).
 */
export function factorImpacts(prediction) {
  const total = prediction.base_value + prediction.factors.reduce((s, f) => s + f.contribution, 0);
  return prediction.factors.map((f) => ({
    ...f,
    impact: sigmoid(total) - sigmoid(total - f.contribution), // > 0 : favorise le domicile
  }));
}

function strength(impact) {
  const a = Math.abs(impact);
  if (a >= 0.08) return "fort";
  if (a >= 0.03) return "net";
  if (a >= 0.01) return "léger";
  return null;
}

const points = (impact) => `${impact >= 0 ? "+" : "−"}${Math.abs(impact * 100).toFixed(1).replace(".", ",")} pts`;

// Barres divergentes : à gauche (bleu) ce qui favorise l'équipe à domicile,
// à droite (rouge) ce qui favorise l'équipe à l'extérieur.
export default function FactorsChart({ prediction, home, away }) {
  const transition = useChartTransition();
  const factors = factorImpacts(prediction);
  const max = Math.max(...factors.map((f) => Math.abs(f.impact)), 0.01);

  return (
    <div>
      <div aria-hidden="true" className="mb-3 grid grid-cols-2 gap-2 text-xs font-semibold text-text-2 sm:ml-48">
        <span className="flex items-center justify-end gap-1.5 text-right">
          <span className="h-2.5 w-2.5 rounded-sm bg-home" /> favorise {home.tricode} (domicile)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm bg-away" /> favorise {away.tricode} (extérieur)
        </span>
      </div>

      <ul className="space-y-3" aria-label="Facteurs de la prédiction, du plus au moins influent">
        {factors.map((f, i) => {
          const s = strength(f.impact);
          const favored = f.impact >= 0 ? home : away;
          const width = `${(Math.abs(f.impact) / max) * 78}%`;
          const summary = s ? `avantage ${favored.tricode}, ${s}` : "neutre";
          return (
            <li key={f.key} className="grid items-center gap-x-4 gap-y-1 sm:grid-cols-[11rem_1fr]">
              <div className="text-sm">
                <span className="font-semibold">{f.label}</span>
                <span className="block text-xs text-text-2">
                  {summary}
                  <span className="sr-only"> ({points(f.impact)})</span>
                </span>
              </div>
              <div aria-hidden="true" className="relative grid h-7 grid-cols-2">
                <span className="absolute inset-y-0 left-1/2 w-px bg-line-strong" />
                <div className="flex items-center justify-end gap-1.5 pr-[2px]">
                  {f.impact > 0 && (
                    <>
                      <span className="text-xs font-semibold tabular-nums">{points(f.impact)}</span>
                      <motion.span
                        className="h-4 rounded-l-[4px] bg-home"
                        initial={{ width: 0 }}
                        animate={{ width }}
                        transition={transition(0.1 + i * 0.06)}
                      />
                    </>
                  )}
                </div>
                <div className="flex items-center gap-1.5 pl-[2px]">
                  {f.impact < 0 && (
                    <>
                      <motion.span
                        className="h-4 rounded-r-[4px] bg-away"
                        initial={{ width: 0 }}
                        animate={{ width }}
                        transition={transition(0.1 + i * 0.06)}
                      />
                      <span className="text-xs font-semibold tabular-nums">{points(-f.impact)}</span>
                    </>
                  )}
                </div>
              </div>
            </li>
          );
        })}
      </ul>

      <DataTable
        caption="Contribution de chaque facteur à la prédiction"
        columns={["Facteur", "Favorise", "Impact sur la proba. domicile", "Contribution (log-odds)"]}
        rows={factors.map((f) => [
          f.label,
          Math.abs(f.impact) < 0.005 ? "—" : f.impact >= 0 ? home.tricode : away.tricode,
          points(f.impact),
          f.contribution.toFixed(3).replace(".", ","),
        ])}
      />
    </div>
  );
}

// Probabilité domicile du modèle comparée à celle de l'Elo seul.
export function ModelVsElo({ prediction, home }) {
  const transition = useChartTransition();
  const rows = [
    { label: "Modèle XGBoost", value: prediction.home_win_probability, cls: "bg-home" },
    { label: "Elo seul", value: prediction.elo_home_probability, cls: "bg-line-strong" },
  ];
  return (
    <div>
      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-2">
        Probabilité de victoire {home.tricode}
      </p>
      <ul className="space-y-2">
        {rows.map((r, i) => (
          <li key={r.label} className="grid grid-cols-[7.5rem_1fr_3rem] items-center gap-3 text-sm">
            <span className="text-text-2">{r.label}</span>
            <span aria-hidden="true" className="relative h-2.5 rounded-full bg-surface-2">
              <motion.span
                className={`absolute inset-y-0 left-0 rounded-full ${r.cls}`}
                initial={{ width: 0 }}
                animate={{ width: `${r.value * 100}%` }}
                transition={transition(0.2 + i * 0.1)}
              />
              <span className="absolute -inset-y-1 left-1/2 w-px bg-text-2" />
            </span>
            <span className="text-right font-semibold tabular-nums">{pct(r.value)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
