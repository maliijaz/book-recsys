import Link from "next/link";
import { listPersonas } from "@/lib/api-client";
import { BookCard } from "@/components/BookCard";

export const dynamic = "force-dynamic";

export default async function PersonasPage() {
  const personas = await listPersonas();

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold">Sample personas</h1>
        <p className="mt-1 text-sm" style={{ color: "var(--muted)" }}>
          Real readers from the dataset, with recommendations precomputed offline by the
          trained hybrid model — this is the batch-serving path.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {personas.map((persona) => (
          <Link
            key={persona.persona_id}
            href={`/personas/${persona.persona_id}`}
            className="card rounded-lg p-4"
          >
            <p className="mb-3 text-sm font-medium">Persona #{persona.persona_id}</p>
            <div className="grid grid-cols-3 gap-2">
              {persona.favorite_books.slice(0, 3).map((book) => (
                <BookCard key={book.book_id} book={book} linkToDetail={false} />
              ))}
            </div>
            <p className="mt-3 text-xs" style={{ color: "var(--muted)" }}>
              See recommendations →
            </p>
          </Link>
        ))}
      </div>
    </div>
  );
}
