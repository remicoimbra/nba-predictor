"use client";

import { motion } from "motion/react";
import { pct } from "@/lib/format";
import { useChartTransition } from "@/lib/charts";

const SIZE = 184;
const STROKE = 14;
const R = (SIZE - STROKE) / 2;

// Jauge circulaire : probabilité de victoire de l'équipe favorite, dans la
// couleur de son camp (domicile bleu / extérieur rouge), sur une piste neutre.
export default function WinGauge({ probability, team, side }) {
  const transition = useChartTransition();

  return (
    <div
      className="relative mx-auto"
      style={{ width: SIZE, height: SIZE }}
      role="img"
      aria-label={`${team.name} favori, ${pct(probability)} de probabilité de victoire`}
    >
      <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`} className="-rotate-90" aria-hidden="true">
        <circle cx={SIZE / 2} cy={SIZE / 2} r={R} fill="none" stroke="var(--surface-2)" strokeWidth={STROKE} />
        <motion.circle
          cx={SIZE / 2}
          cy={SIZE / 2}
          r={R}
          fill="none"
          stroke={side === "home" ? "var(--home)" : "var(--away)"}
          strokeWidth={STROKE}
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: probability }}
          transition={transition(0.15, 1.1)}
        />
      </svg>
      <div aria-hidden="true" className="absolute inset-0 flex flex-col items-center justify-center text-center">
        <span className="text-5xl font-bold tracking-tight">{pct(probability)}</span>
        <span className="mt-1 text-sm font-semibold text-text-2">
          victoire <span className="text-text">{team.tricode}</span>
        </span>
      </div>
    </div>
  );
}
