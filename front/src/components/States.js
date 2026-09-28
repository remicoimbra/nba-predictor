// États de chargement / erreur / vide, communs aux pages.

export function CardSkeletons({ count = 6 }) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3" aria-hidden="true">
      {Array.from({ length: count }, (_, i) => (
        <div key={i} className="rounded-2xl border border-line p-5">
          <div className="skeleton mb-5 h-4 w-32 rounded" />
          <div className="flex justify-between">
            <div className="skeleton h-12 w-12 rounded-2xl" />
            <div className="skeleton h-12 w-12 rounded-2xl" />
          </div>
          <div className="skeleton mt-3 h-4 w-full rounded" />
          <div className="skeleton mt-6 h-2.5 w-full rounded-full" />
          <div className="skeleton mt-4 h-16 w-full rounded-xl" />
        </div>
      ))}
    </div>
  );
}

export function Loading({ label = "Chargement…" }) {
  return (
    <p role="status" className="flex items-center gap-3 text-text-2">
      <span aria-hidden="true" className="h-5 w-5 animate-spin rounded-full border-2 border-line border-t-accent" />
      {label}
    </p>
  );
}

export function ErrorState({ error, onRetry, title = "Impossible de charger les données" }) {
  return (
    <div role="alert" className="rounded-2xl border border-line-strong bg-surface p-6">
      <p className="font-display text-xl font-bold text-danger-text">{title}</p>
      <p className="mt-1 text-sm text-text-2">{error?.message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-4 rounded-full bg-accent px-4 py-2 text-sm font-semibold text-on-accent transition-transform hover:scale-[1.03] active:scale-95"
        >
          Réessayer
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, children }) {
  return (
    <div className="flex flex-col items-center rounded-2xl border border-dashed border-line-strong px-6 py-12 text-center">
      <svg viewBox="0 0 64 64" className="h-14 w-14" aria-hidden="true">
        <circle cx="32" cy="32" r="26" fill="var(--surface-2)" />
        <path
          d="M32 6v52M6 32h52M14 13c8 7 8 31 0 38M50 13c-8 7-8 31 0 38"
          fill="none"
          stroke="var(--line-strong)"
          strokeWidth="2.5"
        />
      </svg>
      <p className="mt-4 font-display text-2xl font-bold">{title}</p>
      <div className="mt-1 text-text-2">{children}</div>
    </div>
  );
}
