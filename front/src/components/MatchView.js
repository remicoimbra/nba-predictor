"use client";

import Link from "next/link";
import { motion } from "motion/react";
import { useParams, useSearchParams } from "next/navigation";
import { useId } from "react";
import ComparisonBars from "@/components/charts/ComparisonBars";
import EloLine from "@/components/charts/EloLine";
import FactorsChart, { ModelVsElo } from "@/components/charts/FactorsChart";
import FormChart from "@/components/charts/FormChart";
import WinGauge from "@/components/charts/WinGauge";
import { EmptyState, ErrorState, Loading } from "@/components/States";
import { FormPills, PreseasonBadge, TeamBadge } from "@/components/TeamBits";
import { EASE_OUT } from "@/lib/charts";
import { int, localTime, longDate, record } from "@/lib/format";
import { useApi } from "@/lib/useApi";
import { getGame } from "@/services/api";

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

export default function MatchView() {
  const { id } = useParams();
  const param = useSearchParams().get("date");
  const date = param && ISO_DATE.test(param) ? param : null;
  const game = useApi(`${id}|${date}`, () => getGame(id, date));

  const backHref = date ? `/?date=${date}` : "/";

  return (
    <div className="space-y-6">
      <Link href={backHref} className="inline-flex items-center gap-1.5 text-sm font-semibold text-accent hover:underline">
        <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true">
          <path d="M19 12H5M11 6l-6 6 6 6" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
        </svg>
        Retour aux matchs{date ? ` du ${longDate(date)}` : ""}
      </Link>

      {game.loading && <Loading label="Chargement de l'analyse…" />}
      {game.error &&
        (game.error.status === 404 ? (
          <EmptyState title="Match introuvable">
            <p>{game.error.message}</p>
          </EmptyState>
        ) : (
          <ErrorState error={game.error} onRetry={game.retry} />
        ))}
      {game.data && <MatchDetail game={game.data} />}
    </div>
  );
}

function Section({ title, subtitle, children, index = 0 }) {
  const headingId = useId();
  return (
    <motion.section
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.5, delay: index * 0.05, ease: EASE_OUT }}
      className="rounded-2xl border border-line bg-bg p-5 shadow-card sm:p-6"
      aria-labelledby={headingId}
    >
      <h2 id={headingId} className="font-display text-2xl font-bold uppercase">
        {title}
      </h2>
      {subtitle && <p className="mb-5 mt-1 text-sm text-text-2">{subtitle}</p>}
      {!subtitle && <div className="mb-5" />}
      {children}
    </motion.section>
  );
}

function MatchDetail({ game }) {
  const { home_team: home, away_team: away, prediction } = game;
  const time = localTime(game.game_time_utc);
  const hasStats = Boolean(home.stats && away.stats);
  const favoriteSide = prediction?.predicted_winner;
  const favorite = favoriteSide === "home" ? home : away;
  // Date passée interrogée avec l'état ACTUEL des équipes (team_state.json
  // n'est pas historisé) : des matchs « récents » peuvent être postérieurs.
  const pastDate = [home, away].some((t) => t.recent_games?.some((g) => g.date >= game.date));

  return (
    <div className="space-y-6">
      <h1 className="sr-only">
        {home.name} contre {away.name}, {longDate(game.date)}
      </h1>

      {/* En-tête : équipes + jauge */}
      <motion.div
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, ease: EASE_OUT }}
        className="relative overflow-hidden rounded-3xl border border-line bg-surface p-6 sm:p-8"
      >
        <div aria-hidden="true" className="absolute inset-x-0 top-0 flex h-1.5">
          <span className="flex-1 bg-home" />
          <span className="flex-1 bg-away" />
        </div>
        <div className="mb-6 flex flex-wrap items-center justify-center gap-2 text-sm text-text-2">
          <span className="first-letter:uppercase">{longDate(game.date)}</span>
          {time && <span>· {time}</span>}
          <span>· {game.status}</span>
          {game.season_type === "preseason" && <PreseasonBadge />}
        </div>

        <div className="grid items-center gap-6 md:grid-cols-[1fr_auto_1fr]">
          <HeroTeam team={home} side="home" />
          <div className="order-first md:order-none">
            {prediction ? (
              <>
                <WinGauge probability={Math.max(prediction.home_win_probability, prediction.away_win_probability)} team={favorite} side={favoriteSide} />
                <p className="mt-3 text-center text-sm text-text-2">Score estimé</p>
                <p className="text-center font-display text-4xl font-bold">
                  {int(prediction.predicted_home_score)} <span className="text-text-2">–</span> {int(prediction.predicted_away_score)}
                </p>
              </>
            ) : (
              <p className="max-w-56 text-center text-text-2">Prédiction indisponible : une des équipes est inconnue du modèle.</p>
            )}
          </div>
          <HeroTeam team={away} side="away" />
        </div>

        {prediction && (
          <div className="mx-auto mt-8 max-w-md">
            <ModelVsElo prediction={prediction} home={home} />
          </div>
        )}
      </motion.div>

      {prediction && (
        <Section
          title="Pourquoi cette prédiction ?"
          subtitle="Poids de chaque famille de statistiques dans la décision du modèle, en points de probabilité (contributions XGBoost de type SHAP)."
        >
          <FactorsChart prediction={prediction} home={home} away={away} />
        </Section>
      )}

      {hasStats && (
        <Section title="Face-à-face" subtitle="Les statistiques utilisées par le modèle, côte à côte." index={1}>
          <ComparisonBars home={home} away={away} />
        </Section>
      )}

      {hasStats && (
        <Section title="Forme récente" subtitle="Écart de points sur les 10 derniers matchs : au-dessus de zéro, une victoire." index={2}>
          <div className="grid gap-8 md:grid-cols-2">
            <FormChart team={home} side="home" />
            <FormChart team={away} side="away" />
          </div>
        </Section>
      )}

      {hasStats && (
        <Section title="Évolution Elo" subtitle="Rating après chaque match. Plus il est haut, plus l'équipe est forte." index={3}>
          <EloLine home={home} away={away} />
        </Section>
      )}

      <Section title="À savoir" index={4}>
        <ul className="list-disc space-y-2 pl-5 text-sm text-text-2">
          <li>
            Le modèle prédit uniquement le vainqueur. Le score est une estimation simple, tirée de l&apos;attaque et de la défense
            récentes des deux équipes.
          </li>
          {game.season_type === "preseason" && (
            <li>Match de présaison : le modèle a été entraîné sur la saison régulière, et les rotations sont expérimentales.</li>
          )}
          {(home.new_season || away.new_season) && (
            <li>Nouvelle saison : les bilans repartent de zéro et l&apos;Elo est ramené de 25 % vers la moyenne (1500).</li>
          )}
          {pastDate && (
            <li>
              Date passée : les statistiques affichées sont celles d&apos;aujourd&apos;hui, pas celles du jour du match (pas
              d&apos;historique par date).
            </li>
          )}
        </ul>
      </Section>
    </div>
  );
}

function HeroTeam({ team, side }) {
  const end = side === "away";
  return (
    <div className={`flex flex-col items-center gap-2 text-center ${end ? "md:items-end md:text-right" : "md:items-start md:text-left"}`}>
      <TeamBadge team={team} side={side} size="lg" />
      <p className="font-display text-3xl font-bold uppercase leading-none">{team.name}</p>
      <p className="text-sm text-text-2">
        {side === "home" ? "Domicile" : "Extérieur"}
        {team.record && team.record.wins + team.record.losses > 0 && ` · ${record(team.record)}`}
        {team.elo != null && ` · Elo ${int(team.elo)}`}
      </p>
      <FormPills results={team.last_results} align={end ? "end" : "start"} />
    </div>
  );
}
