"use client";

import { motion } from "motion/react";
import { pct } from "@/lib/format";
import { useChartTransition } from "@/lib/charts";

// Barre de probabilité de victoire domicile (bleu, à gauche) / extérieur
// (rouge, à droite), séparées par un espace de 2px, pourcentages écrits.
export default function ProbabilityBar({ home, away, homeProbability, delay = 0, size = "md" }) {
  const transition = useChartTransition();
  const awayProbability = 1 - homeProbability;
  const height = size === "lg" ? "h-3.5" : "h-2.5";

  return (
    <div>
      <div className="flex items-baseline justify-between text-sm">
        <span>
          <span className="font-semibold">{home.tricode}</span>{" "}
          <span className="font-display text-lg font-bold">{pct(homeProbability)}</span>
        </span>
        <span>
          <span className="font-display text-lg font-bold">{pct(awayProbability)}</span>{" "}
          <span className="font-semibold">{away.tricode}</span>
        </span>
      </div>
      <div
        role="img"
        aria-label={`Probabilité de victoire : ${home.name} ${pct(homeProbability)}, ${away.name} ${pct(awayProbability)}`}
        className={`relative mt-1 flex ${height} gap-[2px]`}
      >
        <motion.div
          className="rounded-l-full bg-home"
          initial={{ flexGrow: 1 }}
          animate={{ flexGrow: homeProbability }}
          transition={transition(delay)}
          style={{ flexBasis: 0 }}
        />
        <motion.div
          className="rounded-r-full bg-away"
          initial={{ flexGrow: 1 }}
          animate={{ flexGrow: awayProbability }}
          transition={transition(delay)}
          style={{ flexBasis: 0 }}
        />
        {/* repère 50 % */}
        <span aria-hidden="true" className="absolute -bottom-1 -top-1 left-1/2 w-px bg-text-2" />
      </div>
    </div>
  );
}
