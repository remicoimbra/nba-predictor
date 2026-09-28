"use client";

import { useReducedMotion } from "motion/react";
import { useCallback, useState } from "react";

// Courbe d'animation commune : démarrage vif, arrivée douce.
export const EASE_OUT = [0.22, 1, 0.36, 1];

/**
 * Transition des graphiques. <MotionConfig reducedMotion="user"> ne coupe
 * que les transformations (déplacements, échelles), pas les attributs SVG
 * animés (largeur d'une barre, tracé d'une courbe) : on les coupe ici.
 */
export function useChartTransition() {
  const reduce = useReducedMotion();
  return useCallback(
    (delay = 0, duration = 0.8) => (reduce ? { duration: 0 } : { duration, delay, ease: EASE_OUT }),
    [reduce],
  );
}

/**
 * Largeur réelle d'un conteneur, pour dessiner les SVG en pixels (un
 * viewBox étiré ferait varier la taille du texte avec la largeur d'écran).
 * S'utilise comme ref callback : <div ref={ref}>.
 */
export function useWidth() {
  const [width, setWidth] = useState(0);
  const ref = useCallback((node) => {
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

// Échelle linéaire [d0, d1] -> [r0, r1].
export function scaleLinear([d0, d1], [r0, r1]) {
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  return (v) => r0 + (v - d0) * k;
}

// Graduations « rondes » (1, 2, 5 × 10^n) couvrant [min, max].
export function niceTicks(min, max, count = 4) {
  const span = max - min || 1;
  const raw = span / count;
  const pow = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * pow).find((s) => span / s <= count) ?? 10 * pow;
  const start = Math.floor(min / step) * step;
  const ticks = [];
  for (let v = start; v <= max + step / 2; v += step) ticks.push(Math.round(v * 1e6) / 1e6);
  return ticks;
}

// Libellés français des colonnes de features du modèle (page « Le modèle »).
const FEATURE_BASE_LABELS = {
  elo: "Rating Elo",
  win_pct: "% victoires saison",
  win_pct_context: "% victoires dom./ext.",
  win_pct_last10: "% victoires 10 derniers",
  rest_days: "Jours de repos",
  avg_pts_scored_last5: "Points marqués (5)",
  avg_pts_scored_last10: "Points marqués (10)",
  avg_pts_allowed_last5: "Points encaissés (5)",
  avg_pts_allowed_last10: "Points encaissés (10)",
  net_rtg_last5: "Net Rating (5)",
  net_rtg_last10: "Net Rating (10)",
};
const SIDE_LABELS = { home: "Domicile", away: "Extérieur", diff: "Écart" };

export function featureLabel(column) {
  const [side, ...rest] = column.split("_");
  const base = rest.join("_");
  const label = FEATURE_BASE_LABELS[base] ?? base;
  return SIDE_LABELS[side] ? `${label} · ${SIDE_LABELS[side]}` : column;
}
