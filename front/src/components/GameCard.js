export default function GameCard({ game }) {
  const { home_team, away_team, prediction, season_type } = game;

  return (
    <div className="rounded-xl border border-gray-200 p-4 shadow-sm hover:shadow-md transition-shadow">
      {/* Le modèle n'est entraîné que sur la saison régulière : en
          présaison, les rotations sont expérimentales et la prédiction
          n'est qu'indicative. */}
      {season_type === "preseason" && (
        <div className="flex justify-center mb-2">
          <span
            className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800"
            title="Modèle entraîné sur la saison régulière : prédiction indicative"
          >
            Présaison
          </span>
        </div>
      )}

      <div className="flex justify-between items-center mb-2">
        <span className="font-semibold text-gray-800">{home_team.name}</span>
        <span className="text-gray-400 text-sm">vs</span>
        <span className="font-semibold text-gray-800">{away_team.name}</span>
      </div>

      {/* L'API renvoie prediction: null quand une équipe est absente de
          team_state.json : on affiche quand même le match. */}
      {prediction ? (
        <PredictionDetails
          prediction={prediction}
          home_team={home_team}
          away_team={away_team}
        />
      ) : (
        <p className="text-sm text-center text-gray-500 my-3">
          Prédiction indisponible
        </p>
      )}
    </div>
  );
}

function PredictionDetails({ prediction, home_team, away_team }) {
  const {
    home_win_probability,
    away_win_probability,
    predicted_winner,
    predicted_home_score,
    predicted_away_score,
  } = prediction;

  const winnerName =
    predicted_winner === "home" ? home_team.name : away_team.name;
  const winnerProbPercent = Math.round(
    (predicted_winner === "home"
      ? home_win_probability
      : away_win_probability) * 100,
  );

  return (
    <>
      <div className="text-center text-2xl font-bold text-gray-900 my-3">
        {predicted_home_score.toFixed(1)} - {predicted_away_score.toFixed(1)}
      </div>

      <div className="text-sm text-center text-gray-600">
        Vainqueur prédit :{" "}
        <span className="font-medium text-blue-600">{winnerName}</span> (
        {winnerProbPercent}%)
      </div>
    </>
  );
}
