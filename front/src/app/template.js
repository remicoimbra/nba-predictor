"use client";

import { motion } from "motion/react";
import { EASE_OUT } from "@/lib/charts";

// template.js est remonté à chaque navigation : fondu d'entrée des pages.
export default function Template({ children }) {
  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35, ease: EASE_OUT }}>
      {children}
    </motion.div>
  );
}
