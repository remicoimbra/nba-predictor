"use client";

import { AnimatePresence, motion } from "motion/react";
import { useCallback, useEffect, useRef, useState } from "react";
import CalendarPopover, { Chevron } from "@/components/calendar/CalendarPopover";
import { dayStatus } from "@/components/calendar/dayStatus";
import { addDays, longDate, parseISODate } from "@/lib/format";

const STRIP_DAYS = 14;
const weekdayFmt = new Intl.DateTimeFormat("fr-FR", { weekday: "short" });
const monthShortFmt = new Intl.DateTimeFormat("fr-FR", { month: "short" });

// Premier jour affiché dans le bandeau pour une date donnée : 3 jours avant,
// pour voir aussi les derniers résultats.
const stripStartFor = (iso) => addDays(iso, -3);
const inStrip = (iso, start) => iso >= start && iso < addDays(start, STRIP_DAYS);

/**
 * Sélecteur de date : bandeau de jours défilant + calendrier mensuel.
 * Pastille rouge = jour avec matchs (d'après GET /games/calendar).
 */
export default function DatePicker({ value, today, calendar, onChange }) {
  const [open, setOpen] = useState(false);
  const [stripStart, setStripStart] = useState(() => stripStartFor(value));
  const triggerRef = useRef(null);
  const boundaryRef = useRef(null);
  const stripRef = useRef(null);

  // Date choisie hors du bandeau (via le calendrier) : on recentre, sans
  // état supplémentaire (valeur dérivée).
  const start = inStrip(value, stripStart) ? stripStart : stripStartFor(value);
  const days = Array.from({ length: STRIP_DAYS }, (_, i) => addDays(start, i));

  // Garde le jour choisi visible quand le bandeau défile (petits écrans).
  useEffect(() => {
    stripRef.current
      ?.querySelector(`[data-date="${value}"]`)
      ?.scrollIntoView({ block: "nearest", inline: "center", behavior: "smooth" });
  }, [value, start]);

  const close = useCallback((restoreFocus) => {
    setOpen(false);
    if (restoreFocus) triggerRef.current?.focus();
  }, []);

  function select(iso) {
    onChange(iso);
    setStripStart(stripStartFor(iso));
    close(true);
  }

  return (
    <div className="flex items-stretch gap-2">
      <button
        type="button"
        onClick={() => setStripStart(addDays(start, -7))}
        className="hidden shrink-0 place-items-center rounded-xl border border-line-strong px-2 text-text-2 transition-colors hover:bg-surface hover:text-text sm:grid"
        aria-label="Semaine précédente"
      >
        <Chevron direction="left" />
      </button>

      <div
        ref={stripRef}
        role="group"
        aria-label="Jours"
        className="flex min-w-0 flex-1 snap-x gap-1 overflow-x-auto rounded-2xl bg-surface p-1 [scrollbar-width:none]"
      >
        {days.map((iso) => {
          const d = parseISODate(iso);
          const selected = iso === value;
          const status = dayStatus(calendar, iso);
          return (
            <button
              key={iso}
              data-date={iso}
              type="button"
              aria-pressed={selected}
              aria-current={iso === today ? "date" : undefined}
              aria-disabled={status.disabled || undefined}
              aria-label={`${longDate(iso)}${iso === today ? ", aujourd'hui" : ""}${status.label ? `, ${status.label}` : ""}`}
              onClick={() => !status.disabled && onChange(iso)}
              className={`relative flex min-w-[3.25rem] flex-1 snap-center flex-col items-center rounded-xl px-1 py-2 transition-colors ${
                selected ? "text-on-accent" : status.disabled ? "cursor-not-allowed text-text-2" : "text-text hover:bg-surface-2"
              }`}
            >
              {selected && (
                <motion.span
                  layoutId="strip-selected"
                  aria-hidden="true"
                  className="absolute inset-0 rounded-xl bg-accent shadow-card"
                  transition={{ type: "spring", stiffness: 450, damping: 36 }}
                />
              )}
              <span aria-hidden="true" className={`relative text-[0.7rem] font-semibold uppercase ${selected ? "" : "text-text-2"}`}>
                {weekdayFmt.format(d).replace(".", "")}
              </span>
              <span aria-hidden="true" className="relative font-display text-xl font-bold leading-tight tabular-nums">
                {d.getDate()}
              </span>
              <span
                aria-hidden="true"
                className={`relative text-[0.65rem] ${iso === today ? "font-bold" : ""} ${selected ? "" : iso === today ? "text-accent" : "text-text-2"}`}
              >
                {iso === today
                  ? "auj."
                  : d.getDate() === 1 || iso === days[0]
                    ? monthShortFmt.format(d).replace(".", "")
                    : " "}
              </span>
              <span
                aria-hidden="true"
                className={`relative mt-0.5 h-1.5 w-1.5 rounded-full ${
                  status.count > 0 ? (selected ? "bg-on-accent" : "bg-brand-red") : "bg-transparent"
                }`}
              />
            </button>
          );
        })}
      </div>

      <button
        type="button"
        onClick={() => setStripStart(addDays(start, 7))}
        className="hidden shrink-0 place-items-center rounded-xl border border-line-strong px-2 text-text-2 transition-colors hover:bg-surface hover:text-text sm:grid"
        aria-label="Semaine suivante"
      >
        <Chevron direction="right" />
      </button>

      <div ref={boundaryRef} className="relative shrink-0">
        <button
          ref={triggerRef}
          type="button"
          aria-haspopup="dialog"
          aria-expanded={open}
          onClick={() => (open ? close(false) : setOpen(true))}
          className="flex h-full items-center gap-2 rounded-xl border border-line-strong px-3 text-sm font-semibold transition-colors hover:bg-surface"
        >
          <CalendarIcon />
          <span className="hidden md:inline">Calendrier</span>
          <span className="sr-only md:hidden">Ouvrir le calendrier</span>
        </button>
        <AnimatePresence>
          {open && (
            <CalendarPopover
              value={value}
              today={today}
              calendar={calendar}
              onSelect={select}
              onClose={close}
              boundaryRef={boundaryRef}
            />
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

function CalendarIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="h-5 w-5" aria-hidden="true">
      <rect x="3" y="5" width="18" height="16" rx="3" />
      <path d="M3 10h18M8 3v4M16 3v4" />
    </svg>
  );
}
