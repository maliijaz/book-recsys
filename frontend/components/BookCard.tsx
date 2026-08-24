import Image from "next/image";
import Link from "next/link";
import type { BookSummary } from "@/lib/types";

export function BookCard({
  book,
  selected,
  onSelect,
  linkToDetail = true,
}: {
  book: BookSummary;
  selected?: boolean;
  onSelect?: (book: BookSummary) => void;
  /** Set false when this card is already nested inside another link/button. */
  linkToDetail?: boolean;
}) {
  const content = (
    <div
      className="card flex h-full flex-col gap-2 rounded-lg p-3 transition"
      style={selected ? { outline: "2px solid var(--accent)" } : undefined}
    >
      <div className="relative aspect-[2/3] w-full overflow-hidden rounded" style={{ background: "var(--border)" }}>
        {book.image_url ? (
          <Image src={book.image_url} alt={book.title} fill sizes="200px" className="object-cover" />
        ) : (
          <div className="flex h-full items-center justify-center text-xs" style={{ color: "var(--muted)" }}>
            No cover
          </div>
        )}
      </div>
      <div>
        <p className="line-clamp-2 text-sm font-medium">{book.title}</p>
        <p className="text-xs" style={{ color: "var(--muted)" }}>
          {book.authors}
          {book.average_rating ? ` · ★ ${book.average_rating.toFixed(2)}` : ""}
        </p>
      </div>
    </div>
  );

  if (onSelect) {
    return (
      <button type="button" onClick={() => onSelect(book)} className="text-left">
        {content}
      </button>
    );
  }

  if (!linkToDetail) {
    return content;
  }

  return <Link href={`/books/${book.book_id}`}>{content}</Link>;
}
