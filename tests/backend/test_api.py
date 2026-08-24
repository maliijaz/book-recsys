def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_books(client):
    resp = client.get("/books")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 6
    assert len(body["items"]) == 6


def test_list_books_search_by_title(client):
    resp = client.get("/books", params={"query": "dune"})
    body = resp.json()
    assert body["total"] == 2
    assert all("Dune" in item["title"] for item in body["items"])


def test_list_books_search_by_author(client):
    resp = client.get("/books", params={"query": "austen"})
    body = resp.json()
    assert body["total"] == 2


def test_list_books_pagination(client):
    resp = client.get("/books", params={"page": 1, "page_size": 2})
    body = resp.json()
    assert len(body["items"]) == 2
    assert body["total"] == 6


def test_get_book_detail(client):
    resp = client.get("/books/100")
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "Dune"
    assert body["ratings_count"] == 500


def test_get_book_detail_not_found(client):
    resp = client.get("/books/9999")
    assert resp.status_code == 404


def test_similar_books_surfaces_the_closest_cluster(client):
    resp = client.get("/books/100/similar", params={"k": 1})
    assert resp.status_code == 200
    body = resp.json()
    # book 100 (Dune) was embedded right next to book 101 (Dune Messiah)
    assert body["similar"][0]["book_id"] == 101


def test_similar_books_not_found(client):
    resp = client.get("/books/9999/similar")
    assert resp.status_code == 404


def test_list_personas(client):
    resp = client.get("/personas")
    assert resp.status_code == 200
    body = resp.json()
    assert body[0]["persona_id"] == 1


def test_get_persona_detail(client):
    resp = client.get("/personas/1")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["recommendations"]) == 2


def test_get_persona_not_found(client):
    resp = client.get("/personas/999")
    assert resp.status_code == 404


def test_persona_recommendations_endpoint(client):
    resp = client.get("/personas/1/recommendations")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_live_recommendations_recovers_the_same_cluster(client):
    resp = client.post("/recommendations/live", json={"liked_book_ids": [100], "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["based_on"][0]["book_id"] == 100
    rec_ids = [b["book_id"] for b in body["recommendations"]]
    assert 101 in rec_ids  # Dune Messiah, embedded closest to Dune
    assert 100 not in rec_ids  # never recommend back what was just liked


def test_live_recommendations_unknown_book_id(client):
    resp = client.post("/recommendations/live", json={"liked_book_ids": [9999]})
    assert resp.status_code == 400


def test_live_recommendations_requires_at_least_one_id(client):
    resp = client.post("/recommendations/live", json={"liked_book_ids": []})
    assert resp.status_code == 422  # pydantic min_length validation


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body[0]["model"] == "hybrid"


def test_cors_headers_present_for_allowed_origin(client):
    resp = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"
