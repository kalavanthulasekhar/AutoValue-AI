import unittest
from unittest.mock import patch

import app as api


VALID = {
    "name": "Maruti Swift Dzire VDI", "year": 2017, "km_driven": 60000,
    "fuel": "Diesel", "seller_type": "Individual",
    "transmission": "Manual", "owner": "First Owner",
}


class FakeModel:
    def predict(self, frame):
        return [500000.0]


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = api.app.test_client()
        self.model_patch = patch.object(api, "model", FakeModel())
        self.model_patch.start()

    def tearDown(self):
        self.model_patch.stop()

    def test_health_reports_model_state(self):
        self.assertEqual(self.client.get("/health").json["status"], "healthy")

    def test_prediction_accepts_valid_input(self):
        response = self.client.post("/predict", json=VALID)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["predicted_price"], 500000.0)

    def test_prediction_rejects_invalid_and_missing_input(self):
        response = self.client.post("/predict", json={**VALID, "year": 1800})
        self.assertEqual(response.status_code, 400)
        self.assertIn("year", response.json["error"])
        response = self.client.post("/predict", json={"name": "Car"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("fields", response.json)

    def test_prediction_rejects_non_object_json(self):
        response = self.client.post("/predict", json=["car"])
        self.assertEqual(response.status_code, 400)

    def test_missing_model_returns_service_unavailable(self):
        with patch.object(api, "model", None):
            response = self.client.post("/predict", json=VALID)
            self.assertEqual(response.status_code, 503)
            self.assertEqual(self.client.get("/health").json["status"], "degraded")

    def test_prediction_exception_is_json_server_error(self):
        class BrokenModel:
            def predict(self, frame):
                raise RuntimeError("backend failure")
        with patch.object(api, "model", BrokenModel()):
            response = self.client.post("/predict", json=VALID)
        self.assertEqual(response.status_code, 500)
        self.assertIn("error", response.json)


if __name__ == "__main__":
    unittest.main()
