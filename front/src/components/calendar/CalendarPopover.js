"use client";

import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { EASE_OUT } from "@/lib/charts";
import { addDays, longDate, parseISODate, toISODate } from "@/lib/format";
import { dayStatus } from "@/components/calendar/dayStatus";

const WEEKDAYS = [
  ["lun", "lundi"],
  ["mar", "mardi"],
  ["mer", "mercredi"],
  ["jeu", "jeudi"],
  ["ven", "vendredi"],
  ["sam", "samedi"],
  ["dim", "dimanche"],
];
const monthFmt = new Intl.DateTimeFormat("fr-FR", { month: "long", year: "numeric" });

const monthKey = (iso) => iso.slice(0, 7);

function addMonths(iso, n) {
  const d = parseISODate(iso);
  const day = d.getDate();
  d.setDate(1);
  d.setMonth(d.getMonth() + n);
  // Jour conservé, borné à la fin du mois (31 janv. + 1 mois -> 28/29 févr.)
  const last = new Date(d.getFullYear(), d.getMonth() + 1, 0).getDate();
  d.setDate(Math.min(day, last));
  return toISODate(d);
}

// 6 semaines x 7 jours, en commençant le lundi de la semaine du 1er du mois.
function monthGrid(iso) {
  const first = parseISODate(`${monthKey(iso)}-01`);
  const offset = (first.getDay() + 6) % 7;
  const start = addDays(toISODate(first), -offset);
  return Array.from({ length: 6 }, (_, w) => Array.from({ length: 7 }, (_, d) => addDays(start, w * 7 + d)));
}

/**
 * Grille mensuelle (motif WAI-ARIA « date picker dialog ») :
 * flèches = jour / semaine, Début/Fin = début/fin de semaine,
 * Page préc./suiv. = mois (Maj : année), Entrée = choisir, Échap = fermer.
 */
export default function CalendarPopover({ value, today, calendar, onSelect, onClose, boundaryRef }) {
  const [focused, setFocused] = useState(value);
  const [direction, setDirection] = useState(0);
  const dayRefs = useRef(new Map());
  const panelRef = useRef(null);
  const moveFocusRef = useRef(true); // focus initial sur la date choisie

  const month = monthKey(focused);
  const weeks = monthGrid(focused);

  function moveTo(iso) {
    setDirection(iso > focused ? 1 : iso < focused ? -1 : 0);
    setFocused(iso);
  }

  // Déplace le focus clavier sur le jour actif après une navigation au
  // clavier (pas après un clic sur « mois suivant », pour ne pas voler le
  // focus du bouton).
  useEffect(() => {
    if (!moveFocusRef.current) return;
    moveFocusRef.current = false;
    dayRefs.current.get(focused)?.focus();
  }, [focused]);

  // Fermeture au clic à l'extérieur (`boundaryRef` inclut le bouton
  // d'ouverture : un clic dessus doit refermer, pas fermer puis rouvrir).
  useEffect(() => {
    function onPointerDown(e) {
      const boundary = boundaryRef?.current ?? panelRef.current;
      if (boundary && !boundary.contains(e.target)) onClose(false);
    }
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [onClose, boundaryRef]);

  function onKeyDown(e) {
    const moves = {
      ArrowLeft: () => addDays(focused, -1),
      ArrowRight: () => addDays(focused, 1),
      ArrowUp: () => addDays(focused, -7),
      ArrowDown: () => addDays(focused, 7),
      Home: () => addDays(focused, -((parseISODate(focused).getDay() + 6) % 7)),
      End: () => addDays(focused, 6 - ((parseISODate(focused).getDay() + 6) % 7)),
      PageUp: () => addMonths(focused, e.shiftKey ? -12 : -1),
      PageDown: () => addMonths(focused, e.shiftKey ? 12 : 1),
    };
    if (moves[e.key]) {
      e.preventDefault();
      moveFocusRef.current = true;
      moveTo(moves[e.key]());
    }
  }

  function onPanelKeyDown(e) {
    if (e.key === "Escape") {
      e.stopPropagation();
      onClose(true);
    }
  }

  return (
    <motion.div
      ref={panelRef}
      role="dialog"
      aria-modal="false"
      aria-label="Choisir une date"
      onKeyDown={onPanelKeyDown}
      initial={{ opacity: 0, y: -8, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -8, scale: 0.97 }}
      transition={{ duration: 0.2, ease: EASE_OUT }}
      className="absolute right-0 top-full z-40 mt-2 w-[min(22rem,calc(100vw-2rem))] origin-top-right rounded-2xl border border-line-strong bg-bg p-4 shadow-pop"
    >
      <div className="mb-3 flex items-center justify-between">
        <button
          type="button"
          onClick={() => moveTo(addMonths(focused, -1))}
          className="grid h-9 w-9 place-items-center rounded-full text-text-2 transition-colors hover:bg-surface hover:text-text"
          aria-label="Mois précédent"
        >
          <Chevron direction="left" />
        </button>
        <h2 aria-live="polite" className="font-display text-lg font-bold uppercase tracking-wide">
          {monthFmt.format(parseISODate(focused))}
        </h2>
        <button
          type="button"
          onClick={() => moveTo(addMonths(focused, 1))}
          className="grid h-9 w-9 place-items-center rounded-full text-text-2 transition-colors hover:bg-surface hover:text-text"
          aria-label="Mois suivant"
        >
          <Chevron direction="right" />
        </button>
      </div>

      <div className="overflow-hidden">
        <AnimatePresence mode="popLayout" initial={false} custom={direction}>
          <motion.table
            key={month}
            role="grid"
            aria-label={monthFmt.format(parseISODate(focused))}
            onKeyDown={onKeyDown}
            custom={direction}
            variants={{
              enter: (dir) => ({ x: dir * 40, opacity: 0 }),
              center: { x: 0, opacity: 1 },
              exit: (dir) => ({ x: dir * -40, opacity: 0 }),
            }}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ duration: 0.25, ease: EASE_OUT }}
            className="w-full border-collapse"
          >
            <thead>
              <tr>
                {WEEKDAYS.map(([short, full]) => (
                  <th key={short} scope="col" className="pb-1 text-center text-xs font-semibold uppercase text-text-2">
                    <abbr title={full} className="no-underline">
                      {short}
                    </abbr>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {weeks.map((week) => (
                <tr key={week[0]}>
                  {week.map((iso) => {
                    const inMonth = monthKey(iso) === month;
                    const selected = iso === value;
                    const isToday = iso === today;
                    const status = dayStatus(calendar, iso);
                    return (
                      <td key={iso} role="gridcell" aria-selected={selected} className="p-0.5 text-center">
                        <button
                          ref={(node) => {
                            if (node) dayRefs.current.set(iso, node);
                            else dayRefs.current.delete(iso);
                          }}
                          type="button"
                          tabIndex={iso === focused ? 0 : -1}
                          aria-disabled={status.disabled || undefined}
                          aria-current={isToday ? "date" : undefined}
                          aria-label={`${longDate(iso)}${status.label ? `, ${status.label}` : ""}`}
                          onClick={() => {
                            if (status.disabled) return;
                            onSelect(iso);
                          }}
                          className={`relative mx-auto grid h-10 w-10 place-items-center rounded-full text-sm tabular-nums transition-colors ${
                            selected
                              ? "bg-accent font-bold text-on-accent"
                              : status.disabled
                                ? "cursor-not-allowed text-text-2 line-through"
                                : `hover:bg-surface-2 ${inMonth ? "text-text" : "text-text-2"}`
                          } ${isToday && !selected ? "ring-2 ring-inset ring-accent font-bold" : ""}`}
                        >
                          {parseISODate(iso).getDate()}
                          {status.count > 0 && (
                            <span
                              aria-hidden="true"
                              className={`absolute bottom-1 h-1.5 w-1.5 rounded-full ${selected ? "bg-on-accent" : "bg-brand-red"}`}
                            />
                          )}
                        </button>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </motion.table>
        </AnimatePresence>
      </div>

      <div className="mt-3 flex items-center justify-between border-t border-line pt-3 text-xs text-text-2">
        <span className="flex items-center gap-1.5">
          <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-brand-red" /> jour avec matchs
        </span>
        <button
          type="button"
          onClick={() => onSelect(today)}
          className="rounded-full px-3 py-1.5 font-semibold text-accent transition-colors hover:bg-surface"
        >
          Aujourd&apos;hui
        </button>
      </div>
    </motion.div>
  );
}

export function Chevron({ direction, className = "h-4 w-4" }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" className={className} aria-hidden="true">
      <path d={direction === "left" ? "M15 6l-6 6 6 6" : "M9 6l6 6-6 6"} />
    </svg>
  );
}
