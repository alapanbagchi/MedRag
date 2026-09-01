import type { Metadata } from "next";
import { Archivo, IBM_Plex_Mono } from "next/font/google";
import { AppProvider } from "@/lib/store";
import { AppShell } from "@/components/layout/AppShell";
import "./globals.css";

const archivo = Archivo({
  subsets: ["latin"],
  variable: "--font-archivo",
  display: "swap",
  weight: ["400", "500", "600", "700", "800", "900"],
});

const plex = IBM_Plex_Mono({
  subsets: ["latin"],
  variable: "--font-plex",
  display: "swap",
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "MedPat — Medical Research Intelligence",
  description: "Evidence-aware medical research, accelerated. Ask the literature, watch the research pipeline, inspect every source.",
};

const themeInit = `(function(){try{var t=localStorage.getItem("medpat:theme");var d=t==="light"?"light":"dark";document.documentElement.classList.add(d);}catch(e){document.documentElement.classList.add("dark");}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${archivo.variable} ${plex.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInit }} />
      </head>
      <body>
        <AppProvider>
          <AppShell>{children}</AppShell>
        </AppProvider>
      </body>
    </html>
  );
}
