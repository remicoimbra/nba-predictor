// Couleurs des 30 franchises, relevées dans leurs logos (public/team-logos/,
// variante L) en comptant les pixels :
// - color : la couleur la plus présente du logo, hors blanc, noir et gris
//   (Nets et Spurs, logos noir et blanc : noir) ;
// - alt : la suivante, pour départager deux équipes de couleurs proches.
// Valeurs exactes des SVG (ex. le vert des Celtics, #008348).
export const TEAM_COLORS = {
  1610612737: { color: "#c8102e", alt: null }, // ATL
  1610612738: { color: "#008348", alt: "#a73832" }, // BOS
  1610612739: { color: "#bc945c", alt: "#860038" }, // CLE
  1610612740: { color: "#001c44", alt: "#ba9d69" }, // NOP
  1610612741: { color: "#ce1141", alt: "#000000" }, // CHI
  1610612742: { color: "#0064b1", alt: "#061922" }, // DAL
  1610612743: { color: "#0d2240", alt: "#ffc627" }, // DEN
  1610612744: { color: "#1d428a", alt: "#fdb927" }, // GSW
  1610612745: { color: "#ce0e2d", alt: "#ffd520" }, // HOU
  1610612746: { color: "#12173f", alt: "#c8102e" }, // LAC
  1610612747: { color: "#fab624", alt: "#542c81" }, // LAL
  1610612748: { color: "#98002e", alt: "#f9a01b" }, // MIA
  1610612749: { color: "#00471b", alt: "#eee1c6" }, // MIL
  1610612750: { color: "#1d428a", alt: "#009a44" }, // MIN
  1610612751: { color: "#000000", alt: null }, // BKN
  1610612752: { color: "#f58426", alt: "#1d428a" }, // NYK
  1610612753: { color: "#0050b5", alt: "#061922" }, // ORL
  1610612754: { color: "#fdbb30", alt: "#002d62" }, // IND
  1610612755: { color: "#1d428a", alt: "#c8102e" }, // PHI
  1610612756: { color: "#f9a01b", alt: "#e56020" }, // PHX
  1610612757: { color: "#cf0a2c", alt: "#000000" }, // POR
  1610612758: { color: "#592d81", alt: "#63717a" }, // SAC
  1610612759: { color: "#000000", alt: "#c4ced4" }, // SAS
  1610612760: { color: "#002d62", alt: "#f05133" }, // OKC
  1610612761: { color: "#bd1b21", alt: "#000000" }, // TOR
  1610612762: { color: "#4e008e", alt: null }, // UTA
  1610612763: { color: "#12173f", alt: "#5d76a9" }, // MEM
  1610612764: { color: "#0d2240", alt: "#cf0a2c" }, // WAS
  1610612765: { color: "#c8102e", alt: "#1d428a" }, // DET
  1610612766: { color: "#1d1160", alt: "#00788c" }, // CHA
};

// Fond des cartes de match (--bg de globals.css) dans chaque thème.
const CARD_BG = { light: "#ffffff", dark: "#060a16" };

// Distance OKLab en dessous de laquelle deux aplats côte à côte se confondent
// (ex. ATL #c8102e / DET #c8102e, PHI / GSW, DEN / WAS).
const MIN_DISTANCE = 0.12;

// Couleur de l'équipe, ou null hors des 30 franchises (club étranger).
export function teamColor(team) {
  return TEAM_COLORS[team?.id]?.color ?? null;
}

// Couleurs des deux camps pour un graphique où elles se touchent : chaque
// équipe garde sa couleur, sauf si les deux se confondent ; l'extérieur passe
// alors à sa 2ᵉ couleur, sinon le domicile. Repli sur la palette du site
// (--home / --away) pour une équipe hors NBA.
export function matchupColors(home, away) {
  const h = TEAM_COLORS[home?.id];
  const a = TEAM_COLORS[away?.id];
  if (!h || !a) return { home: h?.color ?? "var(--home)", away: a?.color ?? "var(--away)" };
  const distinct = (x, y) => x && y && distance(x, y) >= MIN_DISTANCE;
  if (distinct(h.color, a.color)) return { home: h.color, away: a.color };
  if (distinct(h.color, a.alt)) return { home: h.color, away: a.alt };
  if (distinct(h.alt, a.color)) return { home: h.alt, away: a.color };
  return { home: h.color, away: "var(--text-2)" };
}

// Liseré à poser autour d'un aplat dont le contraste avec le fond de carte
// est sous 3:1 (WCAG 1.4.11) : navy ou noir en sombre, or ou orange en
// clair. Couleur CSS light-dark(), ou null si l'aplat se voit dans les deux
// thèmes.
export function lowContrastEdge(color) {
  if (!color.startsWith("#")) return null; // variable du site, déjà vérifiée
  const [light, dark] = ["light", "dark"].map((theme) =>
    contrast(color, CARD_BG[theme]) < 3 ? "var(--line-strong)" : "transparent",
  );
  return light === dark && light === "transparent" ? null : `light-dark(${light}, ${dark})`;
}

function linear(hex) {
  return [1, 3, 5].map((i) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
}

function contrast(a, b) {
  const lum = (hex) => {
    const [r, g, bl] = linear(hex);
    return 0.2126 * r + 0.7152 * g + 0.0722 * bl;
  };
  const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

function oklab(hex) {
  const [r, g, b] = linear(hex);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

function distance(a, b) {
  const [x, y] = [oklab(a), oklab(b)];
  return Math.hypot(x[0] - y[0], x[1] - y[1], x[2] - y[2]);
}
