import { Barlow_Condensed, Inter } from "next/font/google";
import Header from "@/components/Header";
import Providers from "@/components/Providers";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

// Police condensée « sportive » pour les titres, tricodes et scores.
const barlow = Barlow_Condensed({
  variable: "--font-barlow",
  subsets: ["latin"],
  weight: ["600", "700"],
});

export const metadata = {
  title: "NBA Predictor",
  description: "Prédictions de matchs NBA (XGBoost + Elo)",
};

// Applique le thème choisi AVANT le premier rendu (pas de flash clair ->
// sombre). Sans choix enregistré : pas d'attribut, le CSS suit le système.
const THEME_SCRIPT = `(function(){try{var t=localStorage.getItem("theme");if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t)}catch(e){}})()`;

export default function RootLayout({ children }) {
  return (
    <html lang="fr" className={`${inter.variable} ${barlow.variable} antialiased`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="flex min-h-dvh flex-col">
        <a
          href="#contenu"
          className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-accent focus:px-4 focus:py-2 focus:text-on-accent"
        >
          Aller au contenu
        </a>
        <Providers>
          <Header />
          <main id="contenu" className="mx-auto w-full max-w-6xl flex-1 px-4 pb-16 pt-6 sm:px-6">
            {children}
          </main>
          <footer className="border-t border-line py-6 text-center text-sm text-text-2">
            Projet portfolio · modèle XGBoost entraîné sur 7 saisons NBA · prédictions indicatives
          </footer>
        </Providers>
      </body>
    </html>
  );
}
