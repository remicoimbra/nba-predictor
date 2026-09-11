export default function GameCard({ game }) {
  const winProbPercent = Math.round(game.win_probability * 100);

  return (
    <div className="rounded-xl border border-gray-200 p-4 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex justify-between items-center mb-2">
        <span className="font-semibold text-gray-800">{game.home_team}</span>
        <span className="text-gray-400 text-sm">vs</span>
        <span className="font-semibold text-gray-800">{game.away_team}</span>
      </div>

      <div className="text-center text-2xl font-bold text-gray-900 my-3">
        {game.predicted_home_score} - {game.predicted_away_score}
      </div>

      <div className="text-sm text-center text-gray-600">
        Vainqueur prédit :{" "}
        <span className="font-medium text-blue-600">
          {game.predicted_winner}
        </span>{" "}
        ({winProbPercent}%)
      </div>
    </div>
  );
}
