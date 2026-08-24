import Link from "next/link";
import { listBooks } from "@/lib/api-client";
import { BookCard } from "@/components/BookCard";
import { SearchBox } from "@/components/SearchBox";

export const dynamic = "force-dynamic";

const PAGE_SIZE = 24;

export default async function CatalogPage({
  searchParams,
}: {
  searchParams: Promise<{ query?: string; page?: string }>;
}) {
  const params = await searchParams;
  const page = Number(params.page ?? "1");
  const query = params.query;

  const { items, total } = await listBooks({ query, page, pageSize: PAGE_SIZE });
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold">Browse the catalog</h1>
        <p className="mt-1 text-sm" style={{ color: "var(--muted)" }}>
          {total.toLocaleString()} books from goodbooks-10k
        </p>
      </div>

      <SearchBox />

      {items.length === 0 ? (
        <p style={{ color: "var(--muted)" }}>No books matched your search.</p>
      ) : (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
          {items.map((book) => (
            <BookCard key={book.book_id} book={book} />
          ))}
        </div>
      )}

      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-4 text-sm">
          {page > 1 && (
            <Link href={`/?${new URLSearchParams({ ...(query && { query }), page: String(page - 1) })}`}>
              ← Previous
            </Link>
          )}
          <span style={{ color: "var(--muted)" }}>
            Page {page} of {totalPages}
          </span>
          {page < totalPages && (
            <Link href={`/?${new URLSearchParams({ ...(query && { query }), page: String(page + 1) })}`}>
              Next →
            </Link>
          )}
        </div>
      )}
    </div>
  );
}
