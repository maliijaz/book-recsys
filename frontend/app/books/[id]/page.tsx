import Image from "next/image";
import { notFound } from "next/navigation";
import { getBook, getSimilarBooks } from "@/lib/api-client";
import { BookCard } from "@/components/BookCard";

export const dynamic = "force-dynamic";

export default async function BookDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const bookId = Number(id);

  let book;
  try {
    book = await getBook(bookId);
  } catch {
    notFound();
  }

  const { similar } = await getSimilarBooks(bookId, 12);

  return (
    <div className="flex flex-col gap-10">
      <div className="flex flex-col gap-6 sm:flex-row">
        <div
          className="relative aspect-[2/3] w-48 shrink-0 overflow-hidden rounded-lg"
          style={{ background: "var(--border)" }}
        >
          {book.image_url ? (
            <Image src={book.image_url} alt={book.title} fill sizes="200px" className="object-cover" />
          ) : null}
        </div>
        <div className="flex flex-col gap-2">
          <h1 className="text-2xl font-semibold">{book.title}</h1>
          <p style={{ color: "var(--muted)" }}>by {book.authors}</p>
          <div className="flex gap-4 text-sm">
            {book.average_rating && <span>★ {book.average_rating.toFixed(2)} average rating</span>}
            {book.ratings_count && <span>{book.ratings_count.toLocaleString()} ratings</span>}
            {book.original_publication_year && <span>{Math.round(book.original_publication_year)}</span>}
          </div>
        </div>
      </div>

      <div>
        <h2 className="mb-4 text-lg font-semibold">Similar books</h2>
        {similar.length === 0 ? (
          <p style={{ color: "var(--muted)" }}>No similar books found.</p>
        ) : (
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
            {similar.map((b) => (
              <BookCard key={b.book_id} book={b} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
