const API_BASE_URL = "http://127.0.0.1:8000";

export async function getTodayGames(date) {
  const url = date
    ? `${API_BASE_URL}/games/today?date=${date}`
    : `${API_BASE_URL}/games/today`;

  const res = await fetch(url);

  if (!res.ok) {
    throw new Error(`Erreur API: ${res.status}`);
  }

  return res.json();
}
