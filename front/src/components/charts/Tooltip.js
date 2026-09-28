"use client";

import { AnimatePresence, motion } from "motion/react";

// Infobulle positionnée en pixels dans le conteneur (relatif) du graphique.
// Complément du survol : chaque valeur reste lisible dans la vue tableau.
export default function Tooltip({ point, width }) {
  return (
    <AnimatePresence>
      {point && (
        <motion.div
          role="presentation"
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.12 }}
          className="pointer-events-none absolute z-10 w-max max-w-[14rem] rounded-lg border border-line-strong bg-bg px-3 py-2 text-xs shadow-pop"
          style={{
            left: Math.min(Math.max(point.x, 80), width - 80),
            top: point.y,
            translate: "-50% calc(-100% - 10px)",
          }}
        >
          {point.content}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
