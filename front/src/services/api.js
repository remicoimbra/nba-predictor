// URL de l'API : injectée AU BUILD par Next.js (NEXT_PUBLIC_*), rebuild
// nécessaire si elle change. Valeur par défaut : l'API locale (uvicorn).
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail || `Erreur API : ${status}`);
    this.status = status;
  }
}

async function request(path) {
  let res;
  try {
    res = await fetch(`${API_BASE_URL}${path}`);
  } catch {
    // Échec réseau (API arrêtée, CORS) : pas de statut HTTP.
    throw new ApiError(0, "API injoignable. Vérifier qu'elle est bien démarrée.");
  }

  if (!res.ok) {
    let detail;
    try {
      detail = (await res.json()).detail;
    } catch {
      // corps non JSON : message générique
    }
    throw new ApiError(res.status, detail);
  }

  return res.json();
}

// Matchs d'une date (YYYY-MM-DD, aujourd'hui si absente) + prédictions.
export function getGames(date) {
  return request(date ? `/games/today?date=${date}` : "/games/today");
}

// Fiche complète d'un match. La date évite à l'API de parcourir tout le calendrier.
export function getGame(gameId, date) {
  const query = date ? `?date=${date}` : "";
  return request(`/games/${encodeURIComponent(gameId)}${query}`);
}

// Dates synchronisées -> nombre de matchs (pastilles du calendrier).
export function getCalendar() {
  return request("/games/calendar");
}

// Métriques du modèle (page « Le modèle »).
export function getModelMetrics() {
  return request("/model/metrics");
}
