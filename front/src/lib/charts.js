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
