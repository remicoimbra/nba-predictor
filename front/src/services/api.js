const API_BASE_URL = "http://127.0.0.1:8000";

export async function getTodayGames() {
  const res = await fetch(`${API_BASE_URL}/games/today`);

  if (!res.ok) {
    throw new Error(`Erreur API: ${res.status}`);
  }

  return res.json();
}
