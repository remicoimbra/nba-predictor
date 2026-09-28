import { Suspense } from "react";
import HomeView from "@/components/HomeView";
import { CardSkeletons } from "@/components/States";

// HomeView lit ?date= (useSearchParams) : rendu côté client, sous Suspense.
export default function Home() {
  return (
    <Suspense fallback={<CardSkeletons />}>
      <HomeView />
    </Suspense>
  );
}
