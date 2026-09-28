// État d'un jour dans le calendrier, d'après GET /games/calendar :
// - date présente : synchronisée, `count` matchs (0 = aucun match) ;
// - date absente : non synchronisée. Choisissable seulement si l'API sait
//   aller chercher le calendrier en direct (live_fallback, dev local) ;
//   sinon elle répondrait 404, on désactive le jour.
export function dayStatus(calendar, iso) {
  if (!calendar) return { count: null, disabled: false, label: "" };
  const count = calendar.dates?.[iso];
  if (count === undefined) {
    return calendar.live_fallback
      ? { count: null, disabled: false, label: "" }
      : { count: null, disabled: true, label: "calendrier non synchronisé" };
  }
  return {
    count,
    disabled: false,
    label: count === 0 ? "aucun match" : `${count} match${count > 1 ? "s" : ""}`,
  };
}
