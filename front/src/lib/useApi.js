"use client";

import { useEffect, useEffectEvent, useState } from "react";

/**
 * Charge une ressource de l'API et la recharge quand `key` change.
 *
 * Le résultat est étiqueté avec la clé (et le numéro d'essai) qui l'a
 * produit : « chargement » se déduit d'un résultat qui ne correspond pas
 * encore à la clé courante, sans setState synchrone dans l'effet (règle
 * react-hooks/set-state-in-effect). Une réponse arrivée pour une clé déjà
 * abandonnée (changements en rafale) est ignorée.
 *
 * `previous` garde les dernières données reçues pendant un rechargement :
 * on peut les afficher atténuées plutôt que de faire clignoter un squelette.
 */
export function useApi(key, fetcher) {
  const [attempt, setAttempt] = useState(0);
  const [result, setResult] = useState(null);
  const load = useEffectEvent(() => fetcher());

  useEffect(() => {
    let cancelled = false;
    load()
      .then((data) => {
        if (!cancelled) setResult({ key, attempt, data, error: null });
      })
      .catch((error) => {
        if (!cancelled) setResult({ key, attempt, data: null, error });
      });
    return () => {
      cancelled = true;
    };
  }, [key, attempt]);

  const loading = result?.key !== key || result?.attempt !== attempt;
  return {
    loading,
    data: loading ? null : result.data,
    error: loading ? null : result.error,
    previous: result?.data ?? null,
    retry: () => setAttempt((n) => n + 1),
  };
}
