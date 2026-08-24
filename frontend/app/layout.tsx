import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Shelf — Book Recommendations",
  description: "A deep-learning book recommender: browse, pick a persona, or build your own taste profile.",
};

const NAV_LINKS = [
  { href: "/", label: "Catalog" },
  { href: "/personas", label: "Personas" },
  { href: "/taste-profile", label: "Taste Profile" },
  { href: "/about", label: "About" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="border-b" style={{ borderColor: "var(--border)" }}>
          <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
            <Link href="/" className="text-lg font-semibold">
              📚 Shelf
            </Link>
            <nav className="flex gap-6 text-sm">
              {NAV_LINKS.map((link) => (
                <Link key={link.href} href={link.href} className="hover:opacity-70">
                  {link.label}
                </Link>
              ))}
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-5xl px-6 py-8">{children}</main>
        <footer className="mx-auto max-w-5xl px-6 py-8 text-sm" style={{ color: "var(--muted)" }}>
          Built on the goodbooks-10k dataset. No accounts, no tracking — recommendations for
          sample personas are precomputed offline; the taste-profile flow re-ranks live from
          your picks in this session only.
        </footer>
      </body>
    </html>
  );
}
