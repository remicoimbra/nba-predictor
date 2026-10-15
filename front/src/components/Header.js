"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "motion/react";
import ThemeToggle from "@/components/ThemeToggle";

const NAV = [
  {
    href: "/",
    label: "Matchs",
    match: (path) => path === "/" || path.startsWith("/match"),
  },
  {
    href: "/modele",
    label: "Le modèle",
    match: (path) => path.startsWith("/modele"),
  },
];

export default function Header() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-30 border-b border-line bg-bg">
      <div className="mx-auto flex h-16 max-w-6xl items-center gap-4 px-4 sm:px-6">
        <Link
          href="/"
          className="flex items-center gap-2.5"
          aria-label="NBA Predictor, accueil"
        >
          <Image
            src="/nba-logo.svg"
            alt=""
            width={17}
            height={40}
            unoptimized
            className="h-10 w-auto rounded-sm"
          />
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
                      className="absolute inset-x-3 -bottom-3.25 h-0.75 rounded-full bg-accent"
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
