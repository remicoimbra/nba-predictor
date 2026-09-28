// Petits éléments d'identité d'équipe, partagés entre cartes et fiche match.

// Pastille du tricode, teintée selon le camp (domicile bleu / extérieur rouge).
export function TeamBadge({ team, side, size = "md" }) {
  const sizes = size === "lg" ? "h-16 w-16 text-2xl" : "h-12 w-12 text-lg";
  return (
    <span
      aria-hidden="true"
      className={`grid shrink-0 place-items-center rounded-2xl font-display font-bold tracking-wide ${sizes} ${
        side === "home" ? "bg-home-wash" : "bg-away-wash"
      }`}
      style={{ boxShadow: `inset 0 -3px 0 ${side === "home" ? "var(--home)" : "var(--away)"}` }}
    >
      {team.tricode}
    </span>
  );
}

// Derniers résultats (plus ancien -> plus récent) : V / D écrits, pas
// seulement colorés.
export function FormPills({ results, align = "start" }) {
  if (!results?.length) return null;
  return (
    <span
      className={`flex gap-0.5 ${align === "end" ? "justify-end" : ""}`}
      aria-label={`${results.length} derniers matchs : ${results.map((w) => (w ? "victoire" : "défaite")).join(", ")}`}
      role="img"
    >
      {results.map((win, i) => (
        <span
          key={i}
          aria-hidden="true"
          className={`grid h-4 w-4 place-items-center rounded text-[0.6rem] font-bold ${
            win ? "bg-accent text-on-accent" : "bg-surface-2 text-danger-text"
          }`}
        >
          {win ? "V" : "D"}
        </span>
      ))}
    </span>
  );
}

export function PreseasonBadge() {
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full border border-line-strong px-2 py-0.5 text-xs font-semibold text-text"
      title="Modèle entraîné sur la saison régulière : prédiction indicative"
    >
      <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" aria-hidden="true">
        <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeWidth="2" />
        <path d="M12 8v5M12 16v.5" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
      </svg>
      Présaison · indicatif
    </span>
  );
}
