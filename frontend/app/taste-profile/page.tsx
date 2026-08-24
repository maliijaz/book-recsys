"use client";

import { useEffect, useState } from "react";
import { getLiveRecommendations, listBooks } from "@/lib/api-client";
import { BookCard } from "@/components/BookCard";
import type { BookSummary, LiveRecommendationResponse } from "@/lib/types";

const COLD_START_WARNING_MS = 4000;

export default function TasteProfilePage() {
  const [candidates, setCandidates] = useState<BookSummary[]>([]);
  const [liked, setLiked] = useState<Map<number, BookSummary>>(new Map());
  const [result, setResult] = useState<LiveRecommendationResponse | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "waking" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    listBooks({ pageSize: 30 })
      .then((res) => setCandidates(res.items))
      .catch(() => setCandidates([]));
  }, []);

  function toggleLike(book: BookSummary) {
    setLiked((prev) => {
      const next = new Map(prev);
      if (next.has(book.book_id)) next.delete(book.book_id);
      else next.set(book.book_id, book);
      return next;
    });
  }

  async function handleSubmit() {
    setStatus("loading");
    setErrorMessage("");
    // Render's free tier spins the backend down after 15 min idle -- if the
    // request takes a while, tell the visitor what's happening instead of
    // letting the UI look stuck.
    const wakingTimer = setTimeout(() => setStatus("waking"), COLD_START_WARNING_MS);
    try {
      const res = await getLiveRecommendations([...liked.keys()], 12);
      setResult(res);
      setStatus("idle");
    } catch (err) {
      setStatus("error");
      setErrorMessage(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      clearTimeout(wakingTimer);
    }
  }

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-2xl font-semibold">Build your taste profile</h1>
        <p className="mt-1 text-sm" style={{ color: "var(--muted)" }}>
          Pick a few books you like below — no account needed. We&apos;ll pool their learned
          embeddings on the fly and re-rank the whole catalog for you, live.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-5">
        {candidates.map((book) => (
          <BookCard
            key={book.book_id}
            book={book}
            selected={liked.has(book.book_id)}
            onSelect={toggleLike}
          />
        ))}
      </div>

      <div className="flex items-center gap-4">
        <button
          type="button"
          disabled={liked.size === 0 || status === "loading" || status === "waking"}
          onClick={handleSubmit}
          className="rounded-md px-5 py-2.5 text-sm font-medium disabled:opacity-40"
          style={{ background: "var(--accent)", color: "var(--accent-foreground)" }}
        >
          Get my recommendations ({liked.size} picked)
        </button>
        {status === "waking" && (
          <span className="text-sm" style={{ color: "var(--muted)" }}>
            Waking up the model server — this can take up to a minute on the first request…
          </span>
        )}
        {status === "error" && <span className="text-sm text-red-500">{errorMessage}</span>}
      </div>

      {result && (
        <div>
          <h2 className="mb-4 text-lg font-semibold">
            Recommended based on {result.based_on.map((b) => b.title).join(", ")}
          </h2>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-5">
            {result.recommendations.map((book) => (
              <BookCard key={book.book_id} book={book} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
