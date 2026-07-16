// Root layout - applies globals.css and site-wide metadata to every page.
// The metadata exported here is the site-wide fallback title/description;
// individual pages override it by exporting their own generateMetadata function.
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Tic-Tac-Go Daily Solver",
  description: "Today's Tic-Tac-Go puzzle, captured automatically and replayed with an optimal solve path.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
