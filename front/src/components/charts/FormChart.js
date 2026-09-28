"use client";

import { motion } from "motion/react";
import { useState } from "react";
import DataTable from "@/components/charts/DataTable";
import Tooltip from "@/components/charts/Tooltip";
import { niceTicks, scaleLinear, useChartTransition, useWidth } from "@/lib/charts";
import { shortDate, signed } from "@/lib/format";

const HEIGHT = 190;
const M = { top: 12, right: 8, bottom: 42, left: 34 };

// Écart de points des derniers matchs d'une équipe : colonne vers le haut
// = victoire, vers le bas = défaite (V/D écrit sous chaque colonne).
export default function FormChart({ team, side }) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState(null);
  const transition = useChartTransition();
  const games = team.recent_games ?? [];
  const color = side === "home" ? "var(--home)" : "var(--away)";

  const margins = games.map((g) => g.pts_scored - g.pts_allowed);
  const extent = Math.max(10, ...margins.map(Math.abs));
  const ticks = niceTicks(-extent, extent, 4);
  const yMax = Math.max(...ticks.map(Math.abs));
  const plotW = Math.max(width - M.left - M.right, 0);
  const plotH = HEIGHT - M.top - M.bottom;
  const y = scaleLinear([-yMax, yMax], [M.top + plotH, M.top]);
  const band = games.length ? plotW / games.length : 0;
  const barW = Math.min(24, band * 0.6);

  const wins = games.filter((g) => g.win).length;
  const summary = `${team.name} : ${wins} victoire${wins > 1 ? "s" : ""} et ${games.length - wins} défaite${
    games.length - wins > 1 ? "s" : ""
  } sur les ${games.length} derniers matchs`;

  return (
    <figure>
      <figcaption className="mb-2 flex items-center gap-2 font-semibold">
        <span aria-hidden="true" className="h-3 w-3 rounded-sm" style={{ background: color }} />
        {team.name}
        <span className="ml-auto text-sm font-normal text-text-2">
          {wins}V – {games.length - wins}D
        </span>
      </figcaption>

      <div ref={ref} className="relative" onMouseLeave={() => setHover(null)}>
        {width > 0 && games.length > 0 && (
          <svg width={width} height={HEIGHT} role="img" aria-label={summary} className="overflow-visible">
            {ticks.map((t) => (
              <g key={t}>
                <line
                  x1={M.left}
                  x2={width - M.right}
                  y1={y(t)}
                  y2={y(t)}
                  stroke={t === 0 ? "var(--line-strong)" : "var(--line)"}
                  strokeWidth="1"
                />
                <text x={M.left - 6} y={y(t)} dy="0.32em" textAnchor="end" className="fill-text-2 text-[11px] tabular-nums">
                  {t > 0 ? `+${t}` : t}
                </text>
              </g>
            ))}

            {games.map((g, i) => {
              const cx = M.left + band * i + band / 2;
              const v = margins[i];
              const top = y(Math.max(v, 0));
              const h = Math.max(Math.abs(y(v) - y(0)), 1);
              const r = Math.min(4, barW / 2, h);
              // Coin arrondi côté valeur uniquement, carré côté ligne zéro.
              const path =
                v >= 0
                  ? `M${cx - barW / 2},${y(0)}V${top + r}q0,-${r} ${r},-${r}h${barW - 2 * r}q${r},0 ${r},${r}V${y(0)}Z`
                  : `M${cx - barW / 2},${y(0)}V${y(v) - r}q0,${r} ${r},${r}h${barW - 2 * r}q${r},0 ${r},-${r}V${y(0)}Z`;
              const content = (
                <>
                  <p className="font-semibold">
                    {shortDate(g.date)} · {g.is_home ? "vs" : "@"} {g.opponent?.tricode ?? "?"}
                  </p>
                  <p className="text-text-2">
                    {g.win ? "Victoire" : "Défaite"} {g.pts_scored}–{g.pts_allowed} ({signed(v, 0)})
                  </p>
                </>
              );
              return (
                <g
                  key={`${g.date}-${i}`}
                  onMouseEnter={() => setHover({ x: cx, y: Math.min(top, y(0)), content })}
                  opacity={hover && hover.x !== cx ? 0.55 : 1}
                  className="transition-opacity"
                >
                  {/* zone de survol plus large que la barre */}
                  <rect x={cx - band / 2} y={M.top} width={band} height={plotH} fill="transparent" />
                  <motion.path
                    d={path}
                    fill={color}
                    style={{ originY: v >= 0 ? 1 : 0 }}
                    initial={{ scaleY: 0 }}
                    whileInView={{ scaleY: 1 }}
                    viewport={{ once: true }}
                    transition={transition(i * 0.04, 0.6)}
                  />
                  <text x={cx} y={HEIGHT - M.bottom + 16} textAnchor="middle" className="fill-text text-[11px] font-bold">
                    {g.win ? "V" : "D"}
                  </text>
                  <text x={cx} y={HEIGHT - M.bottom + 31} textAnchor="middle" className="fill-text-2 text-[10px]">
                    {g.opponent?.tricode ?? ""}
                  </text>
                </g>
              );
            })}
          </svg>
        )}
        {games.length === 0 && <p className="text-sm text-text-2">Aucun match récent connu.</p>}
        <Tooltip point={hover} width={width} />
      </div>

      <DataTable
        caption={`Derniers matchs de ${team.name}`}
        columns={["Date", "Adversaire", "Résultat", "Score", "Écart"]}
        rows={games
          .map((g, i) => [
            shortDate(g.date),
            `${g.is_home ? "vs" : "@"} ${g.opponent?.tricode ?? "?"}`,
            g.win ? "Victoire" : "Défaite",
            `${g.pts_scored}–${g.pts_allowed}`,
            signed(margins[i], 0),
          ])
          .reverse()}
      />
    </figure>
  );
}
