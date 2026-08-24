from fastapi import APIRouter, HTTPException, Query, Request

from app.schemas import BookDetail, BookListResponse, SimilarBooksResponse

router = APIRouter(prefix="/books", tags=["books"])


@router.get("", response_model=BookListResponse)
def list_books(
    request: Request,
    query: str | None = Query(default=None, description="Search title/author"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=24, ge=1, le=100),
):
    store = request.app.state.store
    items, total = store.list_books(query, page, page_size)
    return BookListResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/{book_id}", response_model=BookDetail)
def get_book(book_id: int, request: Request):
    store = request.app.state.store
    book = store.get_book_detail(book_id)
    if book is None:
        raise HTTPException(status_code=404, detail=f"Book {book_id} not found")
    return book


@router.get("/{book_id}/similar", response_model=SimilarBooksResponse)
def get_similar_books(book_id: int, request: Request, k: int = Query(default=10, ge=1, le=50)):
    store = request.app.state.store
    similar = store.similar_books(book_id, k=k)
    if similar is None:
        raise HTTPException(status_code=404, detail=f"Book {book_id} not found")
    return SimilarBooksResponse(book_id=book_id, similar=similar)
