"use client";

import Link from "next/link";
import { motion } from "motion/react";
import ProbabilityBar from "@/components/charts/ProbabilityBar";
import { FormPills, PreseasonBadge, TeamBadge } from "@/components/TeamBits";
import { EASE_OUT } from "@/lib/charts";
import { int, localTime, pct, record } from "@/lib/format";

export default function GameCard({ game, date, index = 0 }) {
  const { home_team: home, away_team: away, prediction, season_type } = game;
  const time = localTime(game.game_time_utc);
  const favorite = prediction && (prediction.predicted_winner === "home" ? home : away);

  return (
    <motion.article
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: index * 0.06, ease: EASE_OUT }}
      whileHover={{ y: -3 }}
      className="group relative flex flex-col rounded-2xl border border-line bg-bg p-5 shadow-card transition-shadow hover:shadow-pop focus-within:shadow-pop"
    >
      <div className="mb-4 flex items-center justify-between gap-2 text-sm text-text-2">
        <span className="tabular-nums">
          {time && <span className="font-semibold text-text">{time}</span>}
          {time && " · "}
          {game.status}
        </span>
        {season_type === "preseason" && <PreseasonBadge />}
      </div>

      <div className="grid grid-cols-[1fr_auto_1fr] items-start gap-3">
        <TeamSide team={home} side="home" />
        <span className="mt-3 font-display text-sm font-bold uppercase text-text-2">vs</span>
        <TeamSide team={away} side="away" />
      </div>

      {prediction ? (
        <>
          <div className="mt-5">
            <ProbabilityBar home={home} away={away} homeProbability={prediction.home_win_probability} delay={0.2 + index * 0.06} />
          </div>
          <dl className="mt-4 grid grid-cols-2 gap-3 rounded-xl bg-surface p-3 text-sm">
            <div>
              <dt className="text-text-2">Score estimé</dt>
              <dd className="font-display text-2xl font-bold">
                {int(prediction.predicted_home_score)} – {int(prediction.predicted_away_score)}
              </dd>
            </div>
            <div className="text-right">
              <dt className="text-text-2">Favori</dt>
              <dd className="font-display text-2xl font-bold">{favorite.tricode}</dd>
              <dd className="text-xs text-text-2">
                Elo seul : {favorite.tricode}{" "}
                {pct(prediction.predicted_winner === "home" ? prediction.elo_home_probability : 1 - prediction.elo_home_probability)}
              </dd>
            </div>
          </dl>
        </>
      ) : (
        <p className="mt-5 rounded-xl bg-surface p-3 text-center text-sm text-text-2">
          Prédiction indisponible : équipe inconnue du modèle.
        </p>
      )}

      <Link
        href={`/match/${game.game_id}?date=${date}`}
        className="mt-4 inline-flex items-center gap-1 self-end text-sm font-semibold text-accent after:absolute after:inset-0 after:rounded-2xl after:content-['']"
      >
        Voir l&apos;analyse
        <span className="sr-only">
          {" "}
          {home.name} contre {away.name}
        </span>
        <svg viewBox="0 0 24 24" className="h-4 w-4 transition-transform group-hover:translate-x-1" aria-hidden="true">
          <path d="M5 12h14M13 6l6 6-6 6" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
        </svg>
      </Link>
    </motion.article>
  );
}

function TeamSide({ team, side }) {
  const end = side === "away";
  return (
    <div className={`flex min-w-0 flex-col gap-1.5 ${end ? "items-end text-right" : "items-start"}`}>
      <TeamBadge team={team} side={side} />
      <p className="text-sm font-semibold leading-tight">{team.name}</p>
      <p className="text-xs text-text-2">
        {side === "home" ? "Domicile" : "Extérieur"}
        {team.record && team.record.wins + team.record.losses > 0 && ` · ${record(team.record)}`}
      </p>
      <FormPills results={team.last_results} align={end ? "end" : "start"} />
    </div>
  );
}
