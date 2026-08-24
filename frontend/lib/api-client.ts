import type {
  BookDetail,
  BookListResponse,
  LiveRecommendationResponse,
  ModelMetrics,
  PersonaDetail,
  PersonaSummary,
  SimilarBooksResponse,
} from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export function listBooks(params: { query?: string; page?: number; pageSize?: number } = {}) {
  const search = new URLSearchParams();
  if (params.query) search.set("query", params.query);
  if (params.page) search.set("page", String(params.page));
  if (params.pageSize) search.set("page_size", String(params.pageSize));
  const qs = search.toString();
  return fetchJson<BookListResponse>(`/books${qs ? `?${qs}` : ""}`, { cache: "no-store" });
}

export function getBook(bookId: number) {
  return fetchJson<BookDetail>(`/books/${bookId}`, { cache: "no-store" });
}

export function getSimilarBooks(bookId: number, k = 10) {
  return fetchJson<SimilarBooksResponse>(`/books/${bookId}/similar?k=${k}`, { cache: "no-store" });
}

export function listPersonas() {
  return fetchJson<PersonaSummary[]>("/personas", { cache: "no-store" });
}

export function getPersona(personaId: number) {
  return fetchJson<PersonaDetail>(`/personas/${personaId}`, { cache: "no-store" });
}

export function getLiveRecommendations(likedBookIds: number[], k = 10) {
  return fetchJson<LiveRecommendationResponse>("/recommendations/live", {
    method: "POST",
    body: JSON.stringify({ liked_book_ids: likedBookIds, k }),
  });
}

export function getMetrics() {
  // Metrics don't change at runtime -- safe to let Next.js cache this at build time.
  return fetchJson<ModelMetrics[]>("/metrics");
}
