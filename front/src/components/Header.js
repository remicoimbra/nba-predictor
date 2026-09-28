"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "motion/react";
import ThemeToggle from "@/components/ThemeToggle";

const NAV = [
  { href: "/", label: "Matchs", match: (path) => path === "/" || path.startsWith("/match") },
  { href: "/modele", label: "Le modèle", match: (path) => path.startsWith("/modele") },
];

export default function Header() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-30 border-b border-line bg-bg">
      {/* Liseré bleu / blanc / rouge */}
      <div aria-hidden="true" className="flex h-1">
        <span className="flex-1 bg-accent" />
        <span className="flex-1 bg-surface-2" />
        <span className="flex-1 bg-brand-red" />
      </div>
      <div className="mx-auto flex h-16 max-w-6xl items-center gap-4 px-4 sm:px-6">
        <Link href="/" className="flex items-center gap-2.5" aria-label="NBA Predictor, accueil">
          <Logo />
          <span className="hidden font-display text-xl font-bold uppercase tracking-wide md:inline">
            NBA <span className="text-accent">Predictor</span>
          </span>
        </Link>

        <nav aria-label="Navigation principale" className="sm:ml-4">
          <ul className="flex items-center gap-1">
            {NAV.map((item) => {
              const active = item.match(pathname);
              return (
                <li key={item.href} className="relative">
                  <Link
                    href={item.href}
                    aria-current={active ? "page" : undefined}
                    className={`block rounded-md px-3 py-2 text-sm font-semibold transition-colors hover:bg-surface ${
                      active ? "text-accent" : "text-text-2"
                    }`}
                  >
                    {item.label}
                  </Link>
                  {active && (
                    <motion.span
                      layoutId="nav-indicator"
                      aria-hidden="true"
                      className="absolute inset-x-3 -bottom-[13px] h-[3px] rounded-full bg-accent"
                    />
                  )}
                </li>
              );
            })}
          </ul>
        </nav>

        <div className="ml-auto">
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}

function Logo() {
  return (
    <svg viewBox="0 0 32 32" className="h-8 w-8" aria-hidden="true">
      <circle cx="16" cy="16" r="14" fill="var(--accent)" />
      <path
        d="M16 2v28M2 16h28M6.5 6.5c4 3.5 4 15.5 0 19M25.5 6.5c-4 3.5-4 15.5 0 19"
        fill="none"
        stroke="var(--on-accent)"
        strokeWidth="1.6"
      />
      <circle cx="25" cy="7" r="4.5" fill="var(--brand-red)" stroke="var(--bg)" strokeWidth="2" />
    </svg>
  );
}
