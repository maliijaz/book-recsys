import { notFound } from "next/navigation";
import { getPersona } from "@/lib/api-client";
import { BookCard } from "@/components/BookCard";

export const dynamic = "force-dynamic";

export default async function PersonaDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;

  let persona;
  try {
    persona = await getPersona(Number(id));
  } catch {
    notFound();
  }

  return (
    <div className="flex flex-col gap-10">
      <div>
        <h1 className="text-2xl font-semibold">Persona #{persona.persona_id}</h1>
        <p className="mt-1 text-sm" style={{ color: "var(--muted)" }}>
          Recommendations below were precomputed offline by the trained hybrid model for this
          reader, based on their rating history in goodbooks-10k.
        </p>
      </div>

      <div>
        <h2 className="mb-4 text-lg font-semibold">Rated highly</h2>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-5">
          {persona.favorite_books.map((book) => (
            <BookCard key={book.book_id} book={book} />
          ))}
        </div>
      </div>

      <div>
        <h2 className="mb-4 text-lg font-semibold">Recommended for this persona</h2>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-5">
          {persona.recommendations.map((book) => (
            <BookCard key={book.book_id} book={book} />
          ))}
        </div>
      </div>
    </div>
  );
}
