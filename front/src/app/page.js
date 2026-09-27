"use client";

import { useState, useEffect } from "react";
import { getTodayGames } from "@/services/api";
import GameCard from "@/components/GameCard";

// Date LOCALE (pas toISOString(), qui renvoie la date UTC : entre minuit
// et 2h en France, ça affichait la veille).
function todayISO() {
  const d = new Date();
  const month = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${d.getFullYear()}-${month}-${day}`;
}

export default function Home() {
  const [date, setDate] = useState(todayISO);
  // Résultat du dernier fetch, étiqueté avec SA date : "chargement" se
  // déduit d'un résultat qui ne correspond pas (encore) à la date choisie,
  // sans setState synchrone dans l'effet (règle react-hooks/set-state-in-effect).
  const [result, setResult] = useState(null);

  useEffect(() => {
    // Ignore la réponse d'une date déjà abandonnée (changements en rafale).
    let cancelled = false;

    getTodayGames(date)
      .then((data) => {
        if (!cancelled) setResult({ date, games: data.games ?? [], error: null });
      })
      .catch((err) => {
        if (!cancelled) setResult({ date, games: [], error: err.message });
      });

    return () => {
      cancelled = true;
    };
  }, [date]);

  const loading = result?.date !== date;
  const games = loading ? [] : result.games;
  const error = loading ? null : result.error;

  return (
    <main className="min-h-screen bg-gray-50 p-8">
      <h1 className="text-3xl font-bold text-gray-900 mb-4">Prédictions NBA</h1>

      <div className="mb-6 flex items-center gap-3">
        <label htmlFor="date-picker" className="text-sm text-gray-600">
          Date :
        </label>
        <input
          id="date-picker"
          type="date"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          className="rounded-md border border-gray-300 px-3 py-1.5 text-sm text-black"
        />
      </div>

      {loading && <p className="text-gray-500">Chargement...</p>}

      {error && (
        <p className="text-red-600">Erreur lors du chargement : {error}</p>
      )}

      {!loading && !error && games.length === 0 && (
        <p className="text-gray-500">Aucun match ce jour-là.</p>
      )}

      {!loading && !error && games.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {games.map((game) => (
            <GameCard key={game.game_id} game={game} />
          ))}
        </div>
      )}
    </main>
  );
}
