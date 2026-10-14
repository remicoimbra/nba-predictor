/** @type {import('next').NextConfig} */
const nextConfig = {
  // Export statique (front/out/) servi tel quel par Caddy sur le VPS : aucun
  // serveur Node en production. Possible car toutes les données viennent de
  // l'API, appelée depuis le navigateur. Conséquence : pas de route
  // dynamique sans liste connue au build, d'où /match?id= et pas /match/[id].
  output: "export",
  // Pas de serveur pour optimiser les images à la demande.
  images: { unoptimized: true },
};

export default nextConfig;
