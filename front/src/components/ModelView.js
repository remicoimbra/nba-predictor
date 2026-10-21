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
          partir des écarts entre les deux équipes (Elo, forme, repos,
          effectif…) calculés avant chaque match. Elle a été préférée à XGBoost, dont les
          probabilités sont moins fiables, et elle est comparée au rating Elo
          seul et à la règle naïve «&nbsp;le domicile gagne toujours&nbsp;».
          Le score et les statistiques des joueurs sont estimés à part, et
          mesurés de la même façon.
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
  const scoreError = m.score_model?.evaluation.methods.find(
    (x) => x.key === "model",
  )?.mae_team;
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

      {m.score_model && <ScoreCard score={m.score_model} period={period} />}

      {m.player_model && <PlayerCard players={m.player_model} period={period} />}

      <Card
        title="Ce qui compte le plus"
        subtitle="Poids moyen de chaque famille de statistiques dans les prédictions (part du total). « Forme récente » réunit le bilan et le Net Rating des derniers matchs ; « Effectif » compare les joueurs du dernier match (Game Score des 8 meilleurs) à l'effectif habituel, pour repérer les absents."
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
              ensuite réentraîné sur toutes les saisons
              {m.train_seasons && ` depuis ${m.train_seasons[0]}`}
              {m.season_half_life &&
                `, les plus anciennes comptant moins (poids divisé par deux toutes les ${int(m.season_half_life)} saisons) car le jeu évolue`}
              .
            </p>
          </div>
          <div>
            <h3 className="mb-1 font-display text-lg font-bold uppercase text-text">
              Les limites
            </h3>
            <p>
              Le modèle ne connaît les absences que par le dernier match joué
              (pas par le rapport officiel des blessures), et ignore les
              matchs sans enjeu. Environ un match sur trois reste mal prédit,
              et le score
              estimé se trompe en moyenne de{" "}
              {scoreError ? `${num(scoreError)} points` : "plusieurs points"}{" "}
              par équipe. Les statistiques des joueurs supposent que ceux du
              dernier match jouent, et varient beaucoup d&apos;un soir à
              l&apos;autre.
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

// Erreur moyenne du score estimé (walk-forward), comparée à la méthode
// précédente (moyennes des 10 derniers matchs) et à la référence naïve,
// et couverture réelle de la fourchette. score.evaluation.methods : model,
// previous, naive (data/models/score_model.py).
function ScoreCard({ score, period }) {
  const methods = Object.fromEntries(
    score.evaluation.methods.map((x) => [x.key, x]),
  );
  const { model, previous, naive } = methods;
  const interval = score.evaluation.interval; // absent des métriques d'avant le 2026-10-21
  const stats = [
    { label: "Par équipe", key: "mae_team" },
    { label: "Sur l'écart", key: "mae_margin" },
    { label: "Sur le total", key: "mae_total" },
  ];

  return (
    <Card
      title="Score estimé"
      subtitle={`Écart moyen entre le score estimé et le score réel, saisons ${period} (prédites sans les avoir vues). Plus bas = mieux.`}
    >
      <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((s) => (
          <div key={s.key} className="rounded-xl border border-line bg-surface p-4">
            <dt className="text-sm text-text-2">{s.label}</dt>
            <dd className="mt-1 text-3xl font-bold tracking-tight">
              {num(model[s.key])} pts
            </dd>
            <dd className="text-sm text-text-2">
              méthode précédente {num(previous[s.key])} · naïf{" "}
              {num(naive[s.key])}
            </dd>
          </div>
        ))}
        {interval && (
          <div className="rounded-xl border border-line bg-surface p-4">
            <dt className="text-sm text-text-2">
              Fourchette à {pct(interval.level)}
            </dt>
            <dd className="mt-1 text-3xl font-bold tracking-tight">
              {pct1(interval.coverage)}
            </dd>
            <dd className="text-sm text-text-2">
              des vrais scores dedans (±{int(score.interval.half_width)} pts)
            </dd>
          </div>
        )}
      </dl>
      <p className="mt-5 max-w-3xl text-sm leading-relaxed text-text-2">
        Le total de points vient de deux notes par équipe, mises à jour après
        chaque match comme l&apos;Elo : son rythme (nombre de possessions de
        ses matchs) et les points marqués par possession dans ses matchs,
        au-dessus ou en dessous de la moyenne de la ligue. Elles tiennent compte
        des adversaires rencontrés, contrairement à la méthode précédente
        (moyennes brutes des 10 derniers matchs). L&apos;écart est déduit de
        la probabilité de victoire : plus elle s&apos;éloigne de 50 %, plus
        l&apos;écart grandit, et le favori du modèle est toujours devant.
        {interval &&
          ` La fourchette affichée sur chaque match (±${int(score.interval.half_width)} points autour de chaque score) est réglée pour contenir le vrai score ${pct(interval.level)} du temps ; sur les saisons testées, c'est le cas ${pct1(interval.coverage)} du temps.`}{" "}
        «&nbsp;Naïf&nbsp;» : le score moyen de la saison précédente pour tous
        les matchs.
      </p>
    </Card>
  );
}

const PLAYER_STATS = [
  { key: "min", label: "Minutes", unit: "min" },
  { key: "pts", label: "Points", unit: "pts" },
  { key: "reb", label: "Rebonds", unit: "" },
  { key: "ast", label: "Passes décisives", unit: "" },
  { key: "stl", label: "Interceptions", unit: "" },
  { key: "blk", label: "Contres", unit: "" },
];

// Erreur moyenne des statistiques projetées des joueurs (walk-forward),
// comparée aux moyennes simples, et couverture réelle des fourchettes.
// players.evaluation : data/models/player_model.py (methods : model,
// formula, season, last10).
function PlayerCard({ players, period }) {
  const evaluation = players.evaluation;
  const methods = Object.fromEntries(evaluation.methods.map((x) => [x.key, x.stats]));
  const { model, formula, season, last10 } = methods;
  const { coverage } = evaluation.interval;
  const participation = evaluation.participation;
  // Deux décimales : à 0,1 près, les petites stats (passes, contres) se
  // confondraient avec les références.
  const value = (v, unit) => `${dec(v, 2)}${unit ? ` ${unit}` : ""}`;

  return (
    <Card
      title="Statistiques des joueurs"
      subtitle={`Écart moyen entre la projection et la vraie valeur, pour les joueurs attendus qui ont joué, saisons ${period} (prédites sans les avoir vues). Plus bas = mieux.`}
    >
      <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {PLAYER_STATS.map((s) => (
          <div key={s.key} className="rounded-xl border border-line bg-surface p-4">
            <dt className="text-sm text-text-2">{s.label}</dt>
            <dd className="mt-1 text-3xl font-bold tracking-tight">
              {value(model[s.key].mae, s.unit)}
            </dd>
            <dd className="text-sm text-text-2">
              moyenne de la saison {dec(season[s.key].mae, 2)} · 10 derniers{" "}
              {dec(last10[s.key].mae, 2)}
            </dd>
            <dd className="text-sm text-text-2">
              fourchette : {pct(coverage[s.key])} des valeurs dedans
            </dd>
          </div>
        ))}
      </dl>
      <p className="mt-5 max-w-3xl text-sm leading-relaxed text-text-2">
        Les joueurs attendus sont ceux du dernier match de leur équipe. Leurs
        minutes viennent d&apos;un modèle appris (XGBoost) sur les saisons
        passées : moyennes récentes, repos, nombre de joueurs disponibles,
        rythme et déséquilibre attendus du match. Chaque statistique est
        ensuite le produit de ces minutes et du rendement du joueur par minute
        (ses derniers matchs, rapprochés de la moyenne de la ligue quand il a
        peu joué), ajusté au rythme du match et, pour les rebonds, passes,
        interceptions et contres, à ce que concède l&apos;adversaire. Sans le
        modèle de minutes, l&apos;erreur sur les minutes passerait de{" "}
        {dec(model.min.mae, 2)} à {dec(formula.min.mae, 2)}. Pour les
        statistiques rares (interceptions, contres), l&apos;écart avec les
        simples moyennes reste minime. Limites :{" "}
        {pct(participation.did_not_play)} des joueurs attendus ne jouent
        finalement pas, et {pct(participation.uncovered_minutes)} des minutes
        vont à des joueurs absents du match précédent (retours de blessure).
      </p>
    </Card>
  );
}
