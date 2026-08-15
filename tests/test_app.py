import app as app_module

def test_health():
    client = app_module.app.test_client()
    response = client.get("/health")
    assert response.status_code == 200

def test_home():
    client = app_module.app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Wine Quality" in response.data
