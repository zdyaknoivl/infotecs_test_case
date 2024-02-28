import unittest
from src.model import Model
from config import *
import os
import numpy as np


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = Model(PROJECT_ROOT_PATH, UNIT_TEST_UUID)

    def test_predict_without_model(self):
        """
        Тестирование метода predict, когда модель не была обучена.
        """
        if os.path.exists(self.model.model_pickle_path):
            os.remove(self.model.model_pickle_path)

        with self.assertRaises(ValueError):
            self.model.predict(self.model.test_data)

    def test_predict_without_preprocessor(self):
        """
        Тестирование метода predict, когда препроцессор не был обучен.
        """
        if os.path.exists(self.model.preprocessor_pickle_path):
            os.remove(self.model.preprocessor_pickle_path)

        with self.assertRaises(ValueError):
            self.model.predict(self.model.test_data)

    def test_predict_with_trained_model(self):
        """
        Тестирование метода predict с обученной моделью и препроцессором.
        """
        self.model.fit()
        self.model.dump()

        try:
            predictions = self.model.predict(self.model.test_data)
            self.assertIsInstance(predictions, np.ndarray)
        finally:
            if os.path.exists(self.model.model_pickle_path):
                os.remove(self.model.model_pickle_path)
            if os.path.exists(self.model.preprocessor_pickle_path):
                os.remove(self.model.preprocessor_pickle_path)

if __name__ == '__main__':
    unittest.main()
