"use client";

import { MotionConfig } from "motion/react";

// reducedMotion="user" : les animations de déplacement/échelle de motion
// sont coupées si le système demande moins d'animations.
export default function Providers({ children }) {
  return <MotionConfig reducedMotion="user">{children}</MotionConfig>;
}
