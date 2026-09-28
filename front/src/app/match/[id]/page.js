import { Suspense } from "react";
import MatchView from "@/components/MatchView";
import { Loading } from "@/components/States";

export const metadata = {
  title: "Analyse du match · NBA Predictor",
};

// MatchView lit l'id et ?date= côté client (useParams / useSearchParams).
export default function MatchPage() {
  return (
    <Suspense fallback={<Loading />}>
      <MatchView />
    </Suspense>
  );
}
