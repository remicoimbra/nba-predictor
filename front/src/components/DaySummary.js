"use client";

import { motion } from "motion/react";
import { EASE_OUT } from "@/lib/charts";
import { pct } from "@/lib/format";

// Quatre chiffres clés de la journée, calculés à partir des prédictions.
export default function DaySummary({ games }) {
  const predicted = games.filter((g) => g.prediction);
  if (predicted.length === 0) return null;

  const favProb = (g) => Math.max(g.prediction.home_win_probability, g.prediction.away_win_probability);
  const favTeam = (g) => (g.prediction.predicted_winner === "home" ? g.home_team : g.away_team);
  const closest = predicted.reduce((a, b) => (favProb(b) < favProb(a) ? b : a));
  const safest = predicted.reduce((a, b) => (favProb(b) > favProb(a) ? b : a));
  const avg = predicted.reduce((s, g) => s + favProb(g), 0) / predicted.length;

  const tiles = [
    { label: "Matchs", value: games.length, detail: `${predicted.length} avec prédiction` },
    { label: "Confiance moyenne", value: pct(avg), detail: "proba. du favori" },
    {
      label: "Match le plus serré",
      value: `${closest.home_team.tricode}–${closest.away_team.tricode}`,
      detail: `${favTeam(closest).tricode} ${pct(favProb(closest))}`,
    },
    {
      label: "Plus gros favori",
      value: favTeam(safest).tricode,
      detail: `${pct(favProb(safest))} contre ${(safest.prediction.predicted_winner === "home" ? safest.away_team : safest.home_team).tricode}`,
    },
  ];

  return (
    <dl className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      {tiles.map((t, i) => (
        <motion.div
          key={t.label}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: i * 0.05, ease: EASE_OUT }}
          className="rounded-2xl border border-line bg-surface p-4"
        >
          <dt className="text-sm text-text-2">{t.label}</dt>
          <dd className="mt-1 font-display text-3xl font-bold">{t.value}</dd>
          <dd className="text-sm text-text-2">{t.detail}</dd>
        </motion.div>
      ))}
    </dl>
  );
}
