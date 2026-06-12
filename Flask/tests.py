import unittest
from api import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Welcome to the API', response.data)

    def test_retrain(self):
        response = self.app.get('/retrain')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Model retrained!', response.data)

    def test_predict(self):
        response = self.app.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'target', response.data)
        self.assertIn(b'datetime', response.data)

if __name__ == '__main__':
    unittest.main()

