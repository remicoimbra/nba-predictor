"use client";

import { motion } from "motion/react";
import { useState } from "react";
import DataTable from "@/components/charts/DataTable";
import Tooltip from "@/components/charts/Tooltip";
import { niceTicks, scaleLinear, useChartTransition, useWidth } from "@/lib/charts";
import { int, parseISODate, shortDate, toISODate } from "@/lib/format";

const HEIGHT = 260;
const M = { top: 16, right: 76, bottom: 28, left: 44 };
const ELO_AVERAGE = 1500;

const time = (iso) => parseISODate(iso).getTime();

// Dernier point d'une série à la date `t` ou avant.
function valueAt(series, t) {
  let last = null;
  for (const p of series) {
    if (time(p.date) <= t) last = p;
    else break;
  }
  return last;
}

// Évolution du rating Elo des deux équipes (après chaque match), sur un axe
// de temps commun. Ligne de référence : moyenne de la ligue (1500).
export default function EloLine({ home, away }) {
  const [ref, width] = useWidth();
  const [hoverT, setHoverT] = useState(null);
  const transition = useChartTransition();

  const series = [
    { team: home, color: "var(--home)", points: home.elo_history ?? [] },
    { team: away, color: "var(--away)", points: away.elo_history ?? [] },
  ].filter((s) => s.points.length > 0);

  if (series.length === 0) {
    return (
      <p className="text-sm text-text-2">
        Historique Elo indisponible (team_state.json à régénérer avec la dernière version de team_state.py).
      </p>
    );
  }

  const all = series.flatMap((s) => s.points);
  const t0 = Math.min(...all.map((p) => time(p.date)));
  const t1 = Math.max(...all.map((p) => time(p.date)));
  const values = [...all.map((p) => p.elo), ELO_AVERAGE];
  const ticks = niceTicks(Math.min(...values) - 10, Math.max(...values) + 10, 4);
  const plotW = Math.max(width - M.left - M.right, 0);
  const x = scaleLinear([t0, t1], [M.left, M.left + plotW]);
  const y = scaleLinear([ticks[0], ticks.at(-1)], [HEIGHT - M.bottom, M.top]);
  const path = (pts) => pts.map((p, i) => `${i ? "L" : "M"}${x(time(p.date)).toFixed(1)},${y(p.elo).toFixed(1)}`).join("");

  // Étiquettes de fin de courbe : seulement si elles ne se chevauchent pas,
  // sinon la légende suffit.
  const ends = series.map((s) => ({ ...s, last: s.points.at(-1) }));
  const endLabels = ends.length < 2 || Math.abs(y(ends[0].last.elo) - y(ends[1].last.elo)) >= 16;

  function onMove(e) {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = Math.min(Math.max(e.clientX - rect.left, M.left), M.left + plotW);
    // Aimante sur la date de match la plus proche.
    const target = t0 + ((px - M.left) / (plotW || 1)) * (t1 - t0);
    const nearest = all.reduce((best, p) => (Math.abs(time(p.date) - target) < Math.abs(best - target) ? time(p.date) : best), t0);
    setHoverT(nearest);
  }

  const hoverPoint =
    hoverT != null
      ? {
          x: x(hoverT),
          y: M.top + 10,
          content: (
            <>
              <p className="font-semibold">{shortDate(toISODate(new Date(hoverT)))}</p>
              {series.map((s) => {
                const p = valueAt(s.points, hoverT);
                return (
                  <p key={s.team.id} className="flex items-center gap-1.5">
                    <span aria-hidden="true" className="h-2 w-2 rounded-full" style={{ background: s.color }} />
                    {s.team.tricode} : {p ? int(p.elo) : "—"}
                  </p>
                );
              })}
            </>
          ),
        }
      : null;

  return (
    <figure>
      <div aria-hidden="true" className="mb-2 flex flex-wrap gap-4 text-sm font-semibold">
        {series.map((s) => (
          <span key={s.team.id} className="flex items-center gap-2">
            <span className="h-0.5 w-5 rounded-full" style={{ background: s.color, height: 3 }} />
            {s.team.name}
          </span>
        ))}
      </div>
      <div ref={ref} className="relative">
        {width > 0 && (
          <svg
            width={width}
            height={HEIGHT}
            role="img"
            aria-label={`Évolution du rating Elo : ${ends
              .map((s) => `${s.team.name} ${int(s.points[0].elo)} → ${int(s.last.elo)}`)
              .join(", ")}`}
            onMouseMove={onMove}
            onMouseLeave={() => setHoverT(null)}
            className="overflow-visible"
          >
            {ticks.map((t) => (
              <g key={t}>
                <line x1={M.left} x2={M.left + plotW} y1={y(t)} y2={y(t)} stroke="var(--line)" strokeWidth="1" />
                <text x={M.left - 8} y={y(t)} dy="0.32em" textAnchor="end" className="fill-text-2 text-[11px] tabular-nums">
                  {int(t)}
                </text>
              </g>
            ))}
            <line
              x1={M.left}
              x2={M.left + plotW}
              y1={y(ELO_AVERAGE)}
              y2={y(ELO_AVERAGE)}
              stroke="var(--line-strong)"
              strokeWidth="1"
            />
            <text x={M.left + 4} y={y(ELO_AVERAGE) - 5} className="fill-text-2 text-[11px]">
              moyenne de la ligue (1500)
            </text>
            {[t0, t1].map((t, i) => (
              <text
                key={t}
                x={x(t)}
                y={HEIGHT - 8}
                textAnchor={i ? "end" : "start"}
                className="fill-text-2 text-[11px]"
              >
                {shortDate(toISODate(new Date(t)))}
              </text>
            ))}

            {series.map((s, i) => (
              <motion.path
                key={s.team.id}
                d={path(s.points)}
                fill="none"
                stroke={s.color}
                strokeWidth="2"
                strokeLinejoin="round"
                strokeLinecap="round"
                initial={{ pathLength: 0 }}
                whileInView={{ pathLength: 1 }}
                viewport={{ once: true }}
                transition={transition(i * 0.15, 1.2)}
              />
            ))}

            {ends.map((s) => (
              <g key={s.team.id}>
                <circle cx={x(time(s.last.date))} cy={y(s.last.elo)} r="5" fill={s.color} stroke="var(--bg)" strokeWidth="2" />
                {endLabels && (
                  <text x={x(time(s.last.date)) + 10} y={y(s.last.elo)} dy="0.32em" className="fill-text text-xs font-semibold">
                    {s.team.tricode} {int(s.last.elo)}
                  </text>
                )}
              </g>
            ))}

            {hoverT != null && (
              <g pointerEvents="none">
                <line x1={x(hoverT)} x2={x(hoverT)} y1={M.top} y2={HEIGHT - M.bottom} stroke="var(--line-strong)" />
                {series.map((s) => {
                  const p = valueAt(s.points, hoverT);
                  return p ? (
                    <circle key={s.team.id} cx={x(time(p.date))} cy={y(p.elo)} r="4.5" fill={s.color} stroke="var(--bg)" strokeWidth="2" />
                  ) : null;
                })}
              </g>
            )}
          </svg>
        )}
        <Tooltip point={hoverPoint} width={width} />
      </div>

      <DataTable
        caption="Rating Elo après chaque match"
        columns={["Date", ...series.map((s) => s.team.tricode)]}
        rows={[...new Set(all.map((p) => p.date))]
          .sort()
          .reverse()
          .map((d) => [
            shortDate(d),
            ...series.map((s) => {
              const p = s.points.find((q) => q.date === d);
              return p ? int(p.elo) : "—";
            }),
          ])}
      />
    </figure>
  );
}
