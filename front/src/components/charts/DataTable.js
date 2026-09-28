// Vue tableau d'un graphique : les valeurs restent lisibles sans la
// couleur, sans survol, et par un lecteur d'écran. Repliée par défaut.
export default function DataTable({ caption, columns, rows }) {
  return (
    <details className="group mt-3 text-sm">
      <summary className="inline-flex cursor-pointer select-none items-center gap-1.5 rounded-md font-semibold text-accent hover:underline">
        <svg viewBox="0 0 24 24" className="h-4 w-4 transition-transform group-open:rotate-90" aria-hidden="true">
          <path d="M9 6l6 6-6 6" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
        </svg>
        Voir les données
      </summary>
      <div className="mt-2 overflow-x-auto rounded-lg border border-line">
        <table className="w-full text-left tabular-nums">
          <caption className="sr-only">{caption}</caption>
          <thead className="bg-surface text-text-2">
            <tr>
              {columns.map((c) => (
                <th key={c} scope="col" className="px-3 py-2 font-semibold">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className="border-t border-line">
                {row.map((cell, j) =>
                  j === 0 ? (
                    <th key={j} scope="row" className="px-3 py-1.5 font-medium">
                      {cell}
                    </th>
                  ) : (
                    <td key={j} className="px-3 py-1.5">
                      {cell}
                    </td>
                  ),
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}
