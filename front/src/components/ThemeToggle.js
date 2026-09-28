"use client";

import { motion } from "motion/react";
import { useSyncExternalStore } from "react";

// Choix de thème : "light" | "dark" | "system" (aucun attribut, le CSS
// suit le système via color-scheme + light-dark()). Même clé localStorage
// que le script inline de layout.js.
const STORAGE_KEY = "theme";
const EVENT = "themechange";

function readTheme() {
  try {
    const t = localStorage.getItem(STORAGE_KEY);
    return t === "light" || t === "dark" ? t : "system";
  } catch {
    return "system";
  }
}

function subscribe(callback) {
  window.addEventListener(EVENT, callback);
  window.addEventListener("storage", callback); // autres onglets
  return () => {
    window.removeEventListener(EVENT, callback);
    window.removeEventListener("storage", callback);
  };
}

function applyTheme(theme) {
  const root = document.documentElement;
  try {
    if (theme === "system") localStorage.removeItem(STORAGE_KEY);
    else localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // stockage indisponible (navigation privée) : le choix vaut pour la session
  }
  if (theme === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", theme);
  window.dispatchEvent(new Event(EVENT));
}

const OPTIONS = [
  { value: "light", label: "Clair", icon: SunIcon },
  { value: "dark", label: "Sombre", icon: MoonIcon },
  { value: "system", label: "Auto", icon: AutoIcon },
];

export default function ThemeToggle() {
  // Côté serveur : "system" (le vrai choix n'est connu que du navigateur).
  const theme = useSyncExternalStore(subscribe, readTheme, () => "system");

  return (
    <div role="group" aria-label="Thème" className="flex rounded-full border border-line-strong p-0.5">
      {OPTIONS.map(({ value, label, icon: Icon }) => {
        const active = theme === value;
        return (
          <button
            key={value}
            type="button"
            aria-pressed={active}
            aria-label={label}
            onClick={() => applyTheme(value)}
            className={`relative flex items-center gap-1.5 rounded-full px-2 py-1.5 text-xs sm:px-2.5 sm:py-1 font-semibold transition-colors ${
              active ? "text-on-accent" : "text-text-2 hover:text-text"
            }`}
          >
            {active && (
              <motion.span
                layoutId="theme-pill"
                aria-hidden="true"
                className="absolute inset-0 rounded-full bg-accent"
                transition={{ type: "spring", stiffness: 500, damping: 38 }}
              />
            )}
            <Icon className="relative h-3.5 w-3.5" />
            <span aria-hidden="true" className="relative hidden sm:inline">
              {label}
            </span>
          </button>
        );
      })}
    </div>
  );
}

function SunIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true" {...props}>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </svg>
  );
}

function MoonIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true" {...props}>
      <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
    </svg>
  );
}

function AutoIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true" {...props}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor" />
    </svg>
  );
}
