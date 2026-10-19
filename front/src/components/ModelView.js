"use client";

import { motion } from "motion/react";
import {
  AccuracyBars,
  CalibrationChart,
  ImportanceBars,
  SeasonTable,
} from "@/components/charts/ModelCharts";
import { ErrorState, Loading } from "@/components/States";
import { EASE_OUT } from "@/lib/charts";
import { dec, int, num, pct, pct1 } from "@/lib/format";
import { useApi } from "@/lib/useApi";
import { getModelMetrics } from "@/services/api";

export default function ModelView() {
  const metrics = useApi("metrics", getModelMetrics);

  return (
    <div className="space-y-8">
      <header className="max-w-3xl">
        <p className="text-sm font-semibold uppercase tracking-wider text-danger-text">
          Sous le capot
        </p>
        <h1 className="font-display text-4xl font-bold uppercase leading-none sm:text-5xl">
          Le modèle
        </h1>
        <p className="mt-3 text-text-2">
          Une régression logistique prédit si l&apos;équipe à domicile gagne, à
          partir des écarts entre les deux équipes (Elo, forme, repos…)
          calculés avant chaque match. Elle a été préférée à XGBoost, dont les
          probabilités sont moins fiables, et elle est comparée au rating Elo
          seul et à la règle naïve «&nbsp;le domicile gagne toujours&nbsp;».
        </p>
      </header>

      {metrics.loading && <Loading />}
      {metrics.error && (
        <ErrorState error={metrics.error} onRetry={metrics.retry} />
      )}
      {metrics.data && <ModelDetail m={metrics.data} />}
    </div>
  );
}

function Card({ title, subtitle, children, className = "" }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.5, ease: EASE_OUT }}
      className={`rounded-2xl border border-line bg-bg p-5 shadow-card sm:p-6 ${className}`}
    >
      <h2 className="font-display text-2xl font-bold uppercase">{title}</h2>
      {subtitle && <p className="mt-1 text-sm text-text-2">{subtitle}</p>}
      <div className="mt-5">{children}</div>
    </motion.section>
  );
}

function ModelDetail({ m }) {
  const served = m.models.find((x) => x.key === m.served_model);
  const elo = m.elo_params; // absent des métriques d'avant le 2026-10-19
  const gain = served.accuracy - m.baseline_accuracy;
  const seasons = m.evaluation.test_seasons;
  const period = `${seasons[0]} → ${seasons.at(-1)}`;
  const trainedAt = new Date(m.trained_at).toLocaleDateString("fr-FR", {
    day: "numeric",
    month: "long",
    year: "numeric",
  });

  const tiles = [
    {
      label: `Accuracy ${served.name}`,
      value: pct1(served.accuracy),
      detail: `saisons ${period}`,
      hero: true,
    },
    {
      label: "Gain vs baseline",
      value: `+${num(gain * 100)} pts`,
      detail: `baseline ${pct1(m.baseline_accuracy)}`,
    },
    {
      label: "Log loss",
      value: dec(served.log_loss),
      detail: `plus bas = mieux (naïf : ${dec(m.baseline.log_loss)})`,
    },
    {
      label: "Score de Brier",
      value: dec(served.brier),
      detail: `plus bas = mieux (naïf : ${dec(m.baseline.brier)})`,
    },
  ];

  return (
    <div className="space-y-6">
      <dl className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {tiles.map((t, i) => (
          <motion.div
            key={t.label}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: i * 0.05, ease: EASE_OUT }}
            className={`rounded-2xl border p-4 ${t.hero ? "border-accent bg-accent text-on-accent" : "border-line bg-surface"}`}
          >
            <dt className={`text-sm ${t.hero ? "" : "text-text-2"}`}>
              {t.label}
            </dt>
            <dd className="mt-1 text-3xl font-bold tracking-tight">
              {t.value}
            </dd>
            <dd className={`text-sm ${t.hero ? "" : "text-text-2"}`}>
              {t.detail}
            </dd>
          </motion.div>
        ))}
      </dl>

      <div className="grid items-start gap-6 lg:grid-cols-2">
        <Card
          title="Précision"
          subtitle={`Part des matchs des saisons ${period} dont le vainqueur est bien prédit.`}
        >
          <AccuracyBars metrics={m} />
        </Card>
        <Card
          title="Calibration"
          subtitle="Quand le modèle annonce 70 %, l'équipe gagne-t-elle vraiment 7 fois sur 10 ?"
        >
          <CalibrationChart metrics={m} />
        </Card>
      </div>

      <Card
        title="Saison par saison"
        subtitle="Chaque saison est prédite par un modèle entraîné uniquement sur les saisons précédentes. Log loss et Brier mesurent la qualité des probabilités : plus bas = mieux."
      >
        <SeasonTable metrics={m} />
      </Card>

      <Card
        title="Ce qui compte le plus"
        subtitle="Poids moyen de chaque famille de statistiques dans les prédictions (part du total). « Forme récente » réunit le bilan et le Net Rating des derniers matchs."
      >
        <ImportanceBars metrics={m} />
      </Card>

      <Card title="Comment ça marche">
        <div className="grid gap-6 text-sm leading-relaxed text-text-2 md:grid-cols-3">
          <div>
            <h3 className="mb-1 font-display text-lg font-bold uppercase text-text">
              Le rating Elo
            </h3>
            <p>
              Chaque équipe a une note qui monte après une victoire et baisse
              après une défaite, d&apos;autant plus que l&apos;adversaire était
              fort et l&apos;écart large.
              {elo && (
                <>
                  {" "}
                  L&apos;équipe qui reçoit part avec un bonus de{" "}
                  {int(elo.home_advantage)} points. Entre deux saisons, la note
                  revient de {pct(elo.season_regression)} vers la moyenne
                  (1500).
                </>
              )}{" "}
              C&apos;est de loin la statistique la plus utile au modèle, et
              elle sert aussi de référence («&nbsp;Elo seul&nbsp;»).
            </p>
          </div>
          <div>
            <h3 className="mb-1 font-display text-lg font-bold uppercase text-text">
              Sans tricher
            </h3>
            <p>
              Toutes les statistiques sont calculées avec les seuls matchs joués{" "}
              <u>avant</u> celui à prédire. Chaque saison de {seasons[0]} à{" "}
              {seasons.at(-1)} est prédite par un modèle qui ne l&apos;a jamais
              vue, entraîné sur les saisons précédentes. Le modèle en ligne est
              ensuite réentraîné sur toutes les saisons.
            </p>
          </div>
          <div>
            <h3 className="mb-1 font-display text-lg font-bold uppercase text-text">
              Les limites
            </h3>
            <p>
              Le modèle ignore les blessures, les transferts et les matchs sans
              enjeu. Il ne prédit que le vainqueur : le score affiché est une
              estimation simple. Environ un match sur trois reste mal prédit.
            </p>
          </div>
        </div>
        <p className="mt-6 text-xs text-text-2">
          Modèle entraîné le {trainedAt}.
        </p>
      </Card>
    </div>
  );
}
