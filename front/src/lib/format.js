// Formatage à la française (espaces insécables, virgule décimale).

const percentFmt = new Intl.NumberFormat("fr-FR", { style: "percent", maximumFractionDigits: 0 });
const percent1Fmt = new Intl.NumberFormat("fr-FR", {
  style: "percent",
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});
const numberFmt = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 1 });
const intFmt = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 });

export const pct = (p) => (p == null ? "—" : percentFmt.format(p));
export const pct1 = (p) => (p == null ? "—" : percent1Fmt.format(p));
export const num = (v) => (v == null ? "—" : numberFmt.format(v));
export const int = (v) => (v == null ? "—" : intFmt.format(v));
export const signed = (v, digits = 1) =>
  v == null ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(digits).replace(".", ",")}`;

export const record = (r) => (r ? `${r.wins}-${r.losses}` : "—");

// Dates ISO "YYYY-MM-DD" interprétées en date LOCALE (new Date("2026-01-15")
// les lirait en UTC, et afficherait la veille à l'ouest de Greenwich).
export function parseISODate(iso) {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, m - 1, d);
}

export function toISODate(date) {
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${date.getFullYear()}-${month}-${day}`;
}

export const todayISO = () => toISODate(new Date());

export function addDays(iso, n) {
  const d = parseISODate(iso);
  d.setDate(d.getDate() + n);
  return toISODate(d);
}

const longDateFmt = new Intl.DateTimeFormat("fr-FR", { weekday: "long", day: "numeric", month: "long", year: "numeric" });
const shortDateFmt = new Intl.DateTimeFormat("fr-FR", { day: "numeric", month: "short" });
const timeFmt = new Intl.DateTimeFormat("fr-FR", { hour: "2-digit", minute: "2-digit" });

export const longDate = (iso) => longDateFmt.format(parseISODate(iso));
export const shortDate = (iso) => shortDateFmt.format(parseISODate(iso));

// Heure du match dans le fuseau du visiteur (l'API donne l'heure UTC).
export function localTime(utc) {
  if (!utc) return null;
  const d = new Date(utc);
  return Number.isNaN(d.getTime()) ? null : timeFmt.format(d);
}
