"use client";

import { motion } from "motion/react";
import { AccuracyBars, CalibrationChart, ImportanceBars } from "@/components/charts/ModelCharts";
import { ErrorState, Loading } from "@/components/States";
import { EASE_OUT } from "@/lib/charts";
import { int, num, pct1 } from "@/lib/format";
import { useApi } from "@/lib/useApi";
import { getModelMetrics } from "@/services/api";

export default function ModelView() {
  const metrics = useApi("metrics", getModelMetrics);

  return (
    <div className="space-y-8">
      <header className="max-w-3xl">
        <p className="text-sm font-semibold uppercase tracking-wider text-danger-text">Sous le capot</p>
        <h1 className="font-display text-4xl font-bold uppercase leading-none sm:text-5xl">Le modèle</h1>
        <p className="mt-3 text-text-2">
          Un classifieur XGBoost prédit si l&apos;équipe à domicile gagne, à partir de statistiques calculées avant chaque
          match. Il est comparé à une régression logistique et à la règle naïve « le domicile gagne toujours ».
        </p>
      </header>

      {metrics.loading && <Loading />}
      {metrics.error && <ErrorState error={metrics.error} onRetry={metrics.retry} />}
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
  const xgb = m.models.find((x) => x.key === "xgboost");
  const gain = xgb.accuracy - m.baseline_accuracy;
  const trainedAt = new Date(m.trained_at).toLocaleDateString("fr-FR", { day: "numeric", month: "long", year: "numeric" });

  const tiles = [
    { label: "Accuracy XGBoost", value: pct1(xgb.accuracy), detail: `saison ${m.test_season}`, hero: true },
    { label: "Gain vs baseline", value: `+${num(gain * 100)} pts`, detail: `baseline ${pct1(m.baseline_accuracy)}` },
    { label: "Log loss", value: xgb.log_loss.toFixed(3).replace(".", ","), detail: "plus bas = mieux" },
    { label: "Matchs de test", value: int(m.test_games), detail: `${int(m.train_games)} pour l'entraînement` },
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
            <dt className={`text-sm ${t.hero ? "" : "text-text-2"}`}>{t.label}</dt>
            <dd className="mt-1 text-3xl font-bold tracking-tight">{t.value}</dd>
            <dd className={`text-sm ${t.hero ? "" : "text-text-2"}`}>{t.detail}</dd>
          </motion.div>
        ))}
      </dl>

      <div className="grid items-start gap-6 lg:grid-cols-2">
        <Card title="Précision" subtitle={`Part des matchs de la saison ${m.test_season} dont le vainqueur est bien prédit.`}>
          <AccuracyBars metrics={m} />
        </Card>
        <Card title="Calibration" subtitle="Quand le modèle annonce 70 %, l'équipe gagne-t-elle vraiment 7 fois sur 10 ?">
          <CalibrationChart metrics={m} />
        </Card>
      </div>

      <Card
        title="Ce qui compte le plus"
        subtitle="Importance des 10 premières statistiques dans XGBoost (part du gain total). « Écart » = domicile moins extérieur."
      >
        <ImportanceBars metrics={m} />
      </Card>

      <Card title="Comment ça marche">
        <div className="grid gap-6 text-sm leading-relaxed text-text-2 md:grid-cols-3">
          <div>
            <h3 className="mb-1 font-display text-lg font-bold uppercase text-text">Le rating Elo</h3>
            <p>
              Chaque équipe a une note qui monte après une victoire et baisse après une défaite, d&apos;autant plus que
              l&apos;adversaire était fort et l&apos;écart large. Entre deux saisons, la note revient de 25 % vers la moyenne
              (1500). C&apos;est de loin la statistique la plus utile au modèle.
            </p>
          </div>
          <div>
            <h3 className="mb-1 font-display text-lg font-bold uppercase text-text">Sans tricher</h3>
            <p>
              Toutes les statistiques sont calculées avec les seuls matchs joués <em>avant</em> celui à prédire. Le modèle est
              entraîné sur les saisons passées et testé sur la dernière saison complète ({m.test_season}), qu&apos;il
              n&apos;a jamais vue.
            </p>
          </div>
          <div>
            <h3 className="mb-1 font-display text-lg font-bold uppercase text-text">Les limites</h3>
            <p>
              Le modèle ignore les blessures, les transferts et les matchs sans enjeu. Il ne prédit que le vainqueur : le score
              affiché est une estimation simple. Environ 3 matchs sur 10 restent mal prédits.
            </p>
          </div>
        </div>
        <p className="mt-6 text-xs text-text-2">Modèle entraîné le {trainedAt}.</p>
      </Card>
    </div>
  );
}
