import { int, num } from "@/lib/format";
import { fillStyle } from "@/lib/teamColors";

// Colonnes du tableau : clés de prediction.players[side][i].stats (API,
// data/models/player_model.py), abréviations françaises.
const COLUMNS = [
  { key: "min", label: "MIN", title: "Minutes", format: int },
  { key: "pts", label: "PTS", title: "Points", format: num },
  { key: "reb", label: "REB", title: "Rebonds", format: num },
  { key: "ast", label: "PD", title: "Passes décisives", format: num },
  { key: "stl", label: "INT", title: "Interceptions", format: num },
  { key: "blk", label: "CTR", title: "Contres", format: num },
];

// Statistiques projetées des joueurs attendus d'une équipe (ceux de son
// dernier match), « s'il joue » : valeur, et en dessous la fourchette à
// 80 %. Sur mobile, seul le tableau défile horizontalement (colonne des
// noms figée), pas la page.
export default function PlayerProjections({ team, players, color }) {
  return (
    <div className="min-w-0">
      <p className="mb-3 flex items-center gap-2 text-sm font-semibold">
        <span aria-hidden="true" className="h-3 w-3 rounded-sm" style={fillStyle(color)} />
        {team.name}
      </p>
      {players.length === 0 ? (
        <p className="text-sm text-text-2">
          Projections disponibles après le premier match de la saison de l&apos;équipe : avant,
          ses joueurs ne sont pas connus.
        </p>
      ) : (
        // relative : les textes sr-only (position absolue) des cellules
        // restent dans le conteneur qui défile, sinon ils élargissent la page.
        <div className="relative overflow-x-auto rounded-xl border border-line">
          <table className="w-full text-left text-sm tabular-nums">
            <caption className="sr-only">
              Statistiques projetées des joueurs de {team.name}, avec leur fourchette à 80 %
            </caption>
            <thead className="bg-surface text-text-2">
              <tr>
                <th scope="col" className="sticky left-0 bg-surface px-3 py-2 font-semibold">
                  Joueur
                </th>
                {COLUMNS.map((c) => (
                  <th key={c.key} scope="col" className="px-2 py-2 text-right font-semibold">
                    <abbr title={c.title} className="no-underline">
                      {c.label}
                    </abbr>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {players.map((p) => (
                <tr key={p.player_id} className="border-t border-line">
                  <th scope="row" className="sticky left-0 whitespace-nowrap bg-bg px-3 py-1.5 font-medium">
                    {p.name}
                  </th>
                  {COLUMNS.map((c) => {
                    const { value, range } = p.stats[c.key];
                    const [low, high] = range;
                    return (
                      <td key={c.key} className="px-2 py-1.5 text-right align-top">
                        <span className="block">{c.format(value)}</span>
                        <span aria-hidden="true" className="block whitespace-nowrap text-xs text-text-2">
                          {low === high ? low : `${low}–${high}`}
                        </span>
                        <span className="sr-only">
                          {low === high ? `, fourchette ${low}` : `, fourchette de ${low} à ${high}`}
                        </span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
