"use client";

// Petits éléments d'identité d'équipe, partagés entre cartes et fiche match.

import Image from "next/image";
import { useState } from "react";
import { teamColor } from "@/lib/teamColors";

// Pastille de l'équipe : son logo (public/team-logos/, `npm run fetch:logos`)
// sur sa couleur (lib/teamColors.js). Sans logo (club étranger en présaison,
// fichier manquant) : tricode sur fond teinté selon le camp.
export function TeamBadge({ team, side, size = "md" }) {
  const [brokenId, setBrokenId] = useState(null);
  const large = size === "lg";
  const sizes = large ? "h-16 w-16 text-2xl" : "h-12 w-12 text-lg";
  const logoPx = large ? 48 : 36;
  const color = teamColor(team);
  const hasLogo = color && brokenId !== team.id;
  return (
    <span
      aria-hidden="true"
      className={`grid shrink-0 place-items-center rounded-2xl font-display font-bold tracking-wide ${sizes} ${
        hasLogo ? "" : side === "home" ? "bg-home-wash" : "bg-away-wash"
      }`}
      style={hasLogo ? { backgroundColor: color } : undefined}
    >
      {hasLogo ? (
        // Variante D (pour fond sombre) dans les deux thèmes : le fond est la
        // couleur de l'équipe, foncée pour la plupart, et la variante L
        // disparaît sur certaines (texte noir des Spurs, Jazz tout violet).
        <Image
          src={`/team-logos/${team.id}-D.svg`}
          alt=""
          width={logoPx}
          height={logoPx}
          unoptimized
          onError={() => setBrokenId(team.id)}
        />
      ) : (
        team.tricode
      )}
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
        <circle
          cx="12"
          cy="12"
          r="9"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        />
        <path
          d="M12 8v5M12 16v.5"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
        />
      </svg>
      Présaison · indicatif
    </span>
  );
}
