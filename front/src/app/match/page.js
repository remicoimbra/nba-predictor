import { Suspense } from "react";
import MatchView from "@/components/MatchView";
import { Loading } from "@/components/States";

export const metadata = {
  title: "Analyse du match · NBA Predictor",
};

// MatchView lit ?id= et ?date= côté client (useSearchParams). Pas de route
// /match/[id] : l'export statique ne connaît pas les ID de match au build.
export default function MatchPage() {
  return (
    <Suspense fallback={<Loading />}>
      <MatchView />
    </Suspense>
  );
}
