from pathlib import Path
from fastapi.testclient import TestClient
from src.api.app import app


def test_dashboard_page_and_static_assets():
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200
    assert "Atlas Trading Console" in page.text
    assert client.get("/dashboard/styles.css").status_code == 200
    assert client.get("/dashboard/app.js").status_code == 200


def test_dashboard_declares_utf8_and_has_no_mojibake():
    dashboard = Path(__file__).resolve().parents[1] / "dashboard"
    html = (dashboard / "index.html").read_text(encoding="utf-8")
    js = (dashboard / "app.js").read_text(encoding="utf-8")
    assert '<meta charset="UTF-8">' in html
    for text in (html, js):
        for corrupted in ("\u00e2\u201a\u00b9", "\u00e2\u20ac", "\u00c3", "\u00c2"):
            assert corrupted not in text
    assert "\u20b90" in html
    assert "const money=n=>`\\u20B9${" in js


def test_dashboard_api_contract():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    status = client.get("/api/status").json()
    assert "trading_mode" in status
    assert "database" in status
    assert isinstance(client.get("/api/positions").json(), list)
    assert isinstance(client.get("/api/orders").json(), list)
    assert isinstance(client.get("/api/events").json(), list)
