"""Pydantic response/request models for the public API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class BookSummary(BaseModel):
    book_id: int
    title: str
    authors: str
    average_rating: float | None = None
    original_publication_year: float | None = None
    image_url: str | None = None


class BookDetail(BookSummary):
    ratings_count: int | None = None
    small_image_url: str | None = None


class BookListResponse(BaseModel):
    items: list[BookSummary]
    total: int
    page: int
    page_size: int


class SimilarBooksResponse(BaseModel):
    book_id: int
    similar: list[BookSummary]


class PersonaSummary(BaseModel):
    persona_id: int
    favorite_books: list[BookSummary]


class PersonaDetail(PersonaSummary):
    recommendations: list[BookSummary]


class LiveRecommendationRequest(BaseModel):
    liked_book_ids: list[int] = Field(..., min_length=1, max_length=50)
    k: int = Field(default=10, ge=1, le=50)


class LiveRecommendationResponse(BaseModel):
    based_on: list[BookSummary]
    recommendations: list[BookSummary]
