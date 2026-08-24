export interface BookSummary {
  book_id: number;
  title: string;
  authors: string;
  average_rating: number | null;
  original_publication_year: number | null;
  image_url: string | null;
}

export interface BookDetail extends BookSummary {
  ratings_count: number | null;
  small_image_url: string | null;
}

export interface BookListResponse {
  items: BookSummary[];
  total: number;
  page: number;
  page_size: number;
}

export interface SimilarBooksResponse {
  book_id: number;
  similar: BookSummary[];
}

export interface PersonaSummary {
  persona_id: number;
  favorite_books: BookSummary[];
}

export interface PersonaDetail extends PersonaSummary {
  recommendations: BookSummary[];
}

export interface LiveRecommendationResponse {
  based_on: BookSummary[];
  recommendations: BookSummary[];
}

export interface ModelMetrics {
  model: string;
  overall: Record<string, Record<string, number>>;
  cold_start: {
    cold_user_metrics?: Record<string, Record<string, number>>;
    cold_item_recall?: Record<string, number>;
    n_cold_users?: number;
    n_cold_items?: number;
  };
}
