"use client";

import { useRouter, useSearchParams } from "next/navigation";
import DatePicker from "@/components/calendar/DatePicker";
import DaySummary from "@/components/DaySummary";
import GameCard from "@/components/GameCard";
import { CardSkeletons, EmptyState, ErrorState } from "@/components/States";
import { getCalendar, getGames } from "@/services/api";
import { longDate, todayISO } from "@/lib/format";
import { useApi } from "@/lib/useApi";

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

export default function HomeView() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const today = todayISO();
  // Date dans l'URL (?date=) : le retour depuis une fiche match retombe
  // sur le bon jour, et un lien vers une journée est partageable.
  const param = searchParams.get("date");
  const date = param && ISO_DATE.test(param) ? param : today;

  const games = useApi(date, () => getGames(date));
  const calendar = useApi("calendar", getCalendar);

  function changeDate(iso) {
    router.replace(iso === today ? "/" : `/?date=${iso}`, { scroll: false });
  }

  // Prochain jour avec des matchs, pour guider depuis une journée vide.
  const nextGameDay = Object.entries(calendar.data?.dates ?? {}).find(([d, n]) => d > date && n > 0)?.[0];

  const shown = games.data ?? (games.loading ? games.previous : null);

  return (
    <div className="space-y-6">
      <div>
        <p className="text-sm font-semibold uppercase tracking-wider text-danger-text">Prédictions NBA</p>
        <h1 className="font-display text-4xl font-bold uppercase leading-none sm:text-5xl">
          {date === today ? "Matchs du jour" : "Matchs du"}
          <span className="mt-1 block text-2xl text-text-2 first-letter:uppercase sm:text-3xl">{longDate(date)}</span>
        </h1>
      </div>

      <DatePicker value={date} today={today} calendar={calendar.data} onChange={changeDate} />

      <section aria-live="polite" aria-busy={games.loading} className="space-y-6">
        {games.loading && !games.previous && <CardSkeletons />}

        {games.error &&
          (games.error.status === 404 ? (
            <EmptyState title="Date non synchronisée">
              <p>Le calendrier de cette date n&apos;a pas encore été récupéré.</p>
            </EmptyState>
          ) : (
            <ErrorState error={games.error} onRetry={games.retry} />
          ))}

        {shown && (
          <div className={`space-y-6 transition-opacity duration-300 ${games.loading ? "opacity-50" : ""}`}>
            {shown.games.length === 0 ? (
              <EmptyState title="Aucun match ce jour-là">
                {nextGameDay ? (
                  <button
                    type="button"
                    onClick={() => changeDate(nextGameDay)}
                    className="mt-3 rounded-full bg-accent px-4 py-2 text-sm font-semibold text-on-accent transition-transform hover:scale-[1.03] active:scale-95"
                  >
                    Prochains matchs : {longDate(nextGameDay)} →
                  </button>
                ) : (
                  <p>Choisissez une autre date dans le calendrier.</p>
                )}
              </EmptyState>
            ) : (
              <>
                <DaySummary games={shown.games} />
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                  {shown.games.map((game, i) => (
                    <GameCard key={game.game_id} game={game} date={shown.date} index={i} />
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
