"""Startup smoke tests — catch import/DB-init failures before they hit production."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestStartup:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_chat_ui_loads(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "Wakeel" in r.text
        assert "text/html" in r.headers["content-type"]

    def test_dashboard_loads(self, client):
        r = client.get("/dashboard")
        assert r.status_code == 200
        assert "Dashboard" in r.text
        assert "text/html" in r.headers["content-type"]

    def test_chat_endpoint_rejects_empty_message(self, client):
        r = client.post("/api/chat", json={"message": ""})
        assert r.status_code == 422

    def test_chat_history_unknown_session(self, client):
        r = client.get("/api/chat/history/nonexistent-session-id")
        assert r.status_code == 200
        assert r.json() == []
