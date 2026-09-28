"use client";

import { motion } from "motion/react";
import { useState } from "react";
import DataTable from "@/components/charts/DataTable";
import Tooltip from "@/components/charts/Tooltip";
import { featureLabel, scaleLinear, useChartTransition, useWidth } from "@/lib/charts";
import { int, pct, pct1 } from "@/lib/format";

// Barres horizontales à une seule série. `highlight` : la barre mise en
// avant (couleur d'accent), les autres restent neutres.
function HBars({ items, max, format, caption, columns }) {
  const transition = useChartTransition();
  return (
    <div>
      <ul className="space-y-2.5">
        {items.map((item, i) => (
          <li key={item.label} className="grid grid-cols-[minmax(0,11rem)_1fr_3.5rem] items-center gap-3 text-sm sm:grid-cols-[minmax(0,14rem)_1fr_3.5rem]">
            <span className={`truncate ${item.highlight ? "font-semibold" : "text-text-2"}`} title={item.label}>
              {item.label}
            </span>
            <span aria-hidden="true" className="h-3.5">
              <motion.span
                className={`block h-full rounded-r-[4px] ${item.highlight ? "bg-home" : "bg-line-strong"}`}
                initial={{ width: 0 }}
                whileInView={{ width: `${(item.value / max) * 100}%` }}
                viewport={{ once: true }}
                transition={transition(i * 0.05)}
              />
            </span>
            <span className={`text-right tabular-nums ${item.highlight ? "font-semibold" : ""}`}>{format(item.value)}</span>
          </li>
        ))}
      </ul>
      <DataTable caption={caption} columns={columns} rows={items.map((it) => [it.label, format(it.value)])} />
    </div>
  );
}

// Accuracy des modèles vs baseline, échelle de 0 à 100 % (une barre qui ne
// partirait pas de zéro exagérerait les écarts).
export function AccuracyBars({ metrics }) {
  const items = [
    ...metrics.models.map((m) => ({ label: m.name, value: m.accuracy, highlight: m.key === "xgboost" })),
    { label: "Baseline « le domicile gagne »", value: metrics.baseline_accuracy },
  ];
  return (
    <HBars
      items={items}
      max={1}
      format={pct1}
      caption="Accuracy sur la saison de test"
      columns={["Modèle", "Accuracy"]}
    />
  );
}

// Importance des features (gain XGBoost normalisé), 10 premières.
export function ImportanceBars({ metrics, top = 10 }) {
  const entries = Object.entries(metrics.feature_importances).slice(0, top);
  const items = entries.map(([col, v], i) => ({ label: featureLabel(col), value: v, highlight: i === 0 }));
  return (
    <HBars
      items={items}
      max={entries[0]?.[1] || 1}
      format={pct1}
      caption="Importance des features dans XGBoost"
      columns={["Feature", "Importance"]}
    />
  );
}

const CAL_H = 300;
const CM = { top: 12, right: 16, bottom: 40, left: 48 };

// Calibration : proba prédite (x) contre taux de victoire réel (y) par
// tranche. Sur la diagonale = le modèle annonce 70 % et l'équipe gagne 70 %
// du temps.
export function CalibrationChart({ metrics }) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState(null);
  const transition = useChartTransition();
  const bins = metrics.calibration;

  const size = Math.min(width, 460);
  const plot = Math.max(size - CM.left - CM.right, 0);
  const x = scaleLinear([0, 1], [CM.left, CM.left + plot]);
  const y = scaleLinear([0, 1], [CM.top + plot, CM.top]);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const line = bins.map((b, i) => `${i ? "L" : "M"}${x(b.mean_predicted)},${y(b.actual_rate)}`).join("");

  return (
    <figure>
      <div aria-hidden="true" className="mb-2 flex flex-wrap gap-4 text-sm font-semibold">
        <span className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full bg-home" /> Modèle XGBoost
        </span>
        <span className="flex items-center gap-2">
          <span className="w-5 bg-line-strong" style={{ height: 2 }} /> Calibration parfaite
        </span>
      </div>
      <div ref={ref} className="relative" onMouseLeave={() => setHover(null)}>
        {width > 0 && (
          <svg
            width={size}
            height={plot + CM.top + CM.bottom}
            role="img"
            aria-label="Courbe de calibration : taux de victoire réel en fonction de la probabilité prédite, proche de la diagonale"
            className="overflow-visible"
          >
            {ticks.map((t) => (
              <g key={t}>
                <line x1={x(0)} x2={x(1)} y1={y(t)} y2={y(t)} stroke="var(--line)" />
                <line x1={x(t)} x2={x(t)} y1={y(0)} y2={y(1)} stroke="var(--line)" />
                <text x={x(0) - 8} y={y(t)} dy="0.32em" textAnchor="end" className="fill-text-2 text-[11px] tabular-nums">
                  {pct(t)}
                </text>
                <text x={x(t)} y={y(0) + 16} textAnchor="middle" className="fill-text-2 text-[11px] tabular-nums">
                  {pct(t)}
                </text>
              </g>
            ))}
            <text x={x(0.5)} y={y(0) + 34} textAnchor="middle" className="fill-text-2 text-xs">
              Probabilité prédite (domicile)
            </text>
            <text
              transform={`translate(${x(0) - 38},${y(0.5)}) rotate(-90)`}
              textAnchor="middle"
              className="fill-text-2 text-xs"
            >
              Taux de victoire réel
            </text>
            <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="var(--line-strong)" strokeWidth="2" />
            <motion.path
              d={line}
              fill="none"
              stroke="var(--home)"
              strokeWidth="2"
              strokeLinejoin="round"
              initial={{ pathLength: 0 }}
              whileInView={{ pathLength: 1 }}
              viewport={{ once: true }}
              transition={transition(0.1, 1.1)}
            />
            {bins.map((b) => (
              <g
                key={b.bin_start}
                onMouseEnter={() =>
                  setHover({
                    x: x(b.mean_predicted),
                    y: y(b.actual_rate),
                    content: (
                      <>
                        <p className="font-semibold">
                          Tranche {pct(b.bin_start)} – {pct(b.bin_end)}
                        </p>
                        <p className="text-text-2">
                          Prédit {pct1(b.mean_predicted)} · réel {pct1(b.actual_rate)}
                        </p>
                        <p className="text-text-2">{int(b.count)} matchs</p>
                      </>
                    ),
                  })
                }
              >
                <circle cx={x(b.mean_predicted)} cy={y(b.actual_rate)} r="12" fill="transparent" />
                <circle cx={x(b.mean_predicted)} cy={y(b.actual_rate)} r="5" fill="var(--home)" stroke="var(--bg)" strokeWidth="2" />
              </g>
            ))}
          </svg>
        )}
        <Tooltip point={hover} width={size} />
      </div>
      <DataTable
        caption="Calibration par tranche de probabilité"
        columns={["Tranche", "Proba. prédite moyenne", "Taux réel", "Matchs"]}
        rows={bins.map((b) => [
          `${pct(b.bin_start)} – ${pct(b.bin_end)}`,
          pct1(b.mean_predicted),
          pct1(b.actual_rate),
          int(b.count),
        ])}
      />
    </figure>
  );
}
