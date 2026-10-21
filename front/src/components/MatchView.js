"use client";

import Link from "next/link";
import { motion } from "motion/react";
import { useSearchParams } from "next/navigation";
import { useId } from "react";
import ComparisonBars from "@/components/charts/ComparisonBars";
import EloLine from "@/components/charts/EloLine";
import FactorsChart, { ModelVsElo } from "@/components/charts/FactorsChart";
import FormChart from "@/components/charts/FormChart";
import WinGauge from "@/components/charts/WinGauge";
import PlayerProjections from "@/components/PlayerProjections";
import { EmptyState, ErrorState, Loading } from "@/components/States";
import { FormPills, PreseasonBadge, TeamBadge } from "@/components/TeamBits";
import { EASE_OUT } from "@/lib/charts";
import { int, localTime, longDate, num, record } from "@/lib/format";
import { fillStyle, matchupColors } from "@/lib/teamColors";
import { useApi } from "@/lib/useApi";
import { getGame } from "@/services/api";

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

export default function MatchView() {
  const params = useSearchParams();
  const id = params.get("id");
  const param = params.get("date");
  const date = param && ISO_DATE.test(param) ? param : null;
  // Sans id (lien tronqué), pas d'appel à l'API : état vide directement.
  const game = useApi(`${id}|${date}`, () =>
    id ? getGame(id, date) : Promise.resolve(null),
  );

  const backHref = date ? `/?date=${date}` : "/";

  return (
    <div className="space-y-6">
      <Link
        href={backHref}
        className="inline-flex items-center gap-1.5 text-sm font-semibold text-accent hover:underline"
      >
        <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true">
          <path
            d="M19 12H5M11 6l-6 6 6 6"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
          />
        </svg>
        Retour aux matchs{date ? ` du ${longDate(date)}` : ""}
      </Link>

      {!id && (
        <EmptyState title="Match introuvable">
          <p>Aucun match indiqué dans l&apos;adresse.</p>
        </EmptyState>
      )}
      {id && game.loading && <Loading label="Chargement de l'analyse…" />}
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
  const colors = matchupColors(home, away);
  // Date passée interrogée avec l'état ACTUEL des équipes (team_state.json
  // n'est pas historisé) : des matchs « récents » peuvent être postérieurs.
  const pastDate = [home, away].some((t) =>
    t.recent_games?.some((g) => g.date >= game.date),
  );

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
          <span className="flex-1" style={fillStyle(colors.home)} />
          <span className="flex-1" style={fillStyle(colors.away)} />
        </div>
        <div className="mb-6 flex flex-wrap items-center justify-center gap-2 text-sm text-text-2">
          <span className="first-letter:uppercase">{longDate(game.date)}</span>
          {time && <span>· {time}</span>}
          <span>· {game.status}</span>
          {game.season_type === "preseason" && <PreseasonBadge />}
        </div>

        <div className="grid items-center gap-6 md:grid-cols-[1fr_auto_1fr]">
          <HeroTeam team={home} side="home" />
          <div className="order-first md:order-0">
            {prediction ? (
              <>
                <WinGauge
                  probability={Math.max(
                    prediction.home_win_probability,
                    prediction.away_win_probability,
                  )}
                  team={favorite}
                  color={favoriteSide === "home" ? colors.home : colors.away}
                />
                <p className="mt-3 text-center text-sm text-text-2">
                  Score estimé
                </p>
                <p className="text-center font-display text-4xl font-bold">
                  {int(prediction.predicted_home_score)}{" "}
                  <span className="text-text-2">–</span>{" "}
                  {int(prediction.predicted_away_score)}
                </p>
                <ScoreRanges
                  prediction={prediction}
                  home={home}
                  away={away}
                  colors={colors}
                />
              </>
            ) : (
              <p className="max-w-56 text-center text-text-2">
                Prédiction indisponible : une des équipes est inconnue du
                modèle.
              </p>
            )}
          </div>
          <HeroTeam team={away} side="away" />
        </div>

        {prediction && (
          <div className="mx-auto mt-8 max-w-md">
            <ModelVsElo prediction={prediction} home={home} color={colors.home} />
          </div>
        )}
      </motion.div>

      {prediction && (
        <Section
          title="Pourquoi cette prédiction ?"
          subtitle="Poids de chaque famille de statistiques dans la décision du modèle, en points de probabilité, par rapport à un match moyen."
        >
          <FactorsChart prediction={prediction} home={home} away={away} colors={colors} />
        </Section>
      )}

      {hasStats && (
        <Section
          title="Face-à-face"
          subtitle="Les statistiques utilisées par le modèle, côte à côte."
          index={1}
        >
          <ComparisonBars home={home} away={away} colors={colors} />
        </Section>
      )}

      {hasStats &&
        (home.absent_players?.length > 0 ||
          away.absent_players?.length > 0) && (
          <Section
            title="Absents au dernier match"
            subtitle="Joueurs habituels (parmi les 8 meilleurs de l'équipe, entrés en jeu dans au moins la moitié des 10 derniers matchs) qui n'ont pas joué le dernier match. Le modèle les suppose encore absents."
            index={2}
          >
            <div className="grid gap-6 md:grid-cols-2">
              <AbsentList team={home} color={colors.home} />
              <AbsentList team={away} color={colors.away} />
            </div>
          </Section>
        )}

      {prediction?.players && (
        <Section
          title="Statistiques des joueurs"
          subtitle="Ce que chaque joueur du dernier match de son équipe devrait faire s'il joue ; en dessous, une fourchette qui contient la vraie valeur environ 8 fois sur 10. Tous ne joueront pas : le total de leurs points dépasse donc le score estimé."
          index={3}
        >
          <div className="grid gap-6 xl:grid-cols-2">
            <PlayerProjections team={home} players={prediction.players.home} color={colors.home} />
            <PlayerProjections team={away} players={prediction.players.away} color={colors.away} />
          </div>
        </Section>
      )}

      {hasStats && (
        <Section
          title="Forme récente"
          subtitle="Écart de points sur les 10 derniers matchs : au-dessus de zéro, une victoire."
          index={4}
        >
          <div className="grid gap-8 md:grid-cols-2">
            <FormChart team={home} color={colors.home} />
            <FormChart team={away} color={colors.away} />
          </div>
        </Section>
      )}

      {hasStats && (
        <Section
          title="Évolution Elo"
          subtitle="Rating après chaque match. Plus il est haut, plus l'équipe est forte."
          index={5}
        >
          <EloLine home={home} away={away} colors={colors} />
        </Section>
      )}

      <Section title="À savoir" index={6}>
        <ul className="list-disc space-y-2 pl-5 text-sm text-text-2">
          <li>
            Le score s&apos;appuie sur le rythme et l&apos;efficacité des deux
            équipes (pour le total de points) et sur la probabilité de
            victoire (pour l&apos;écart). Il reste approximatif : la fourchette
            contient le vrai score de chaque équipe environ 8 fois sur 10.
          </li>
          <li>
            Les absences sont déduites du dernier match joué, pas du rapport
            officiel des blessures : un joueur qui revient ce soir est encore
            compté absent, un forfait de dernière minute est ignoré.
          </li>
          <li>
            Les statistiques des joueurs sont celles attendues s&apos;ils
            jouent : minutes estimées par un modèle appris sur les saisons
            passées, puis rythme de chacun par minute, ajusté au rythme du
            match et à l&apos;adversaire. Un joueur qui revient de blessure
            n&apos;y figure pas.
          </li>
          {game.season_type === "preseason" && (
            <li>
              Match de présaison : le modèle a été entraîné sur la saison
              régulière, et les rotations sont expérimentales.
            </li>
          )}
          {(home.new_season || away.new_season) && (
            <li>
              Nouvelle saison : les bilans repartent de zéro et l&apos;Elo est
              en partie ramené vers la moyenne (1500), comme expliqué sur la
              page «&nbsp;Le modèle&nbsp;».
            </li>
          )}
          {pastDate && (
            <li>
              Date passée : les statistiques affichées sont celles
              d&apos;aujourd&apos;hui, pas celles du jour du match (pas
              d&apos;historique par date).
            </li>
          )}
        </ul>
      </Section>
    </div>
  );
}

// Fourchette à 80 % de chaque score (score_model.interval de l'API), dans
// l'ordre du score affiché au-dessus : domicile puis extérieur.
function ScoreRanges({ prediction, home, away, colors }) {
  const ranges = [
    ["home", home, prediction.predicted_home_score_range, colors.home],
    ["away", away, prediction.predicted_away_score_range, colors.away],
  ];
  if (ranges.some(([, , range]) => !range)) return null; // API d'avant le 2026-10-21
  return (
    <div className="mt-2 text-center text-xs text-text-2">
      <p>Fourchette à 80&nbsp;%</p>
      <p className="mt-1 flex items-center justify-center gap-3 tabular-nums">
        {ranges.map(([side, team, [low, high], color]) => (
          <span key={side} className="inline-flex items-center gap-1.5">
            <span
              aria-hidden="true"
              className="h-2.5 w-2.5 rounded-sm"
              style={fillStyle(color)}
            />
            <span aria-hidden="true">
              {low}–{high}
            </span>
            <span className="sr-only">
              {team.name} : entre {low} et {high} points
            </span>
          </span>
        ))}
      </p>
    </div>
  );
}

function AbsentList({ team, color }) {
  const players = team.absent_players ?? [];
  return (
    <div>
      <p className="mb-3 flex items-center gap-2 text-sm font-semibold">
        <span
          aria-hidden="true"
          className="h-3 w-3 rounded-sm"
          style={fillStyle(color)}
        />
        {team.name}
      </p>
      {players.length === 0 ? (
        <p className="text-sm text-text-2">Aucun joueur habituel absent.</p>
      ) : (
        <ul className="divide-y divide-line rounded-xl border border-line">
          {players.map((p) => (
            <li
              key={p.player_id}
              className="flex items-baseline justify-between gap-3 px-3 py-2 text-sm"
            >
              <span className="font-semibold">{p.name}</span>
              <span className="text-right text-text-2 tabular-nums">
                Game Score {num(p.value)} · {p.games}/{p.of} matchs
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function HeroTeam({ team, side }) {
  const end = side === "away";
  return (
    <div
      className={`flex flex-col items-center gap-2 text-center ${end ? "md:items-end md:text-right" : "md:items-start md:text-left"}`}
    >
      <TeamBadge team={team} side={side} size="lg" />
      <p className="font-display text-3xl font-bold uppercase leading-none">
        {team.name}
      </p>
      <p className="text-sm text-text-2">
        {side === "home" ? "Domicile" : "Extérieur"}
        {team.record &&
          team.record.wins + team.record.losses > 0 &&
          ` · ${record(team.record)}`}
        {team.elo != null && ` · Elo ${int(team.elo)}`}
      </p>
      <FormPills results={team.last_results} align={end ? "end" : "start"} />
    </div>
  );
}
