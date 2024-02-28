import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
from src.data_preprocessor import DataPreprocessor


class Model:
    def __init__(self, root_path: str, uuid: str, **model_kwargs: dict) -> None:
        """
        Инициализация класса Model с заданными параметрами модели. Определяет пути к файлам модели, препроцессора, 
        а также пути к данным и для сохранения результатов.

        :param root_path: str - корневой путь директории проекта, используется для определения путей к данным и результатам.
        :param uuid: str - уникальный идентификатор для создания путей к файлам модели и векторизатора.
        :param model_kwargs: dict - аргументы (ключевые слова), передаваемые в RandomForestClassifier при создании новой модели.
        
        Создает или загружает модель и векторизатор, определяет пути к обучающим, валидационным и тестовым данным,
        а также к файлам результатов валидации, предсказаний и объяснений.
        """
        self.uuid = uuid

        self.root_path = root_path

        self.model_pickle_path = os.path.join(self.root_path, 'cache', f'{self.uuid}_model.pkl')
        self.preprocessor_pickle_path = os.path.join(
            self.root_path, 
            'cache', 
            f'{self.uuid}_preprocessor.pkl'
        )

        self.validation_results_path = os.path.join(self.root_path, 'results', 'validation.txt') 
        self.prediction_results_path = os.path.join(self.root_path, 'results', 'prediction.txt') 
        self.explanaition_path = os.path.join(self.root_path, 'results', 'explain.txt') 

        self.model = (
            RandomForestClassifier(**model_kwargs) 
            if not os.path.exists(self.model_pickle_path) 
            else joblib.load(self.model_pickle_path)
            )  
        self.preprocessor = (
            DataPreprocessor() 
            if not os.path.exists(self.preprocessor_pickle_path) 
            else joblib.load(self.preprocessor_pickle_path)
            )

        self.train_data = os.path.join(self.root_path, 'data', 'train.tsv') 
        self.val_data = os.path.join(self.root_path, 'data', 'val.tsv') 
        self.test_data = os.path.join(self.root_path, 'data', 'test.tsv') 

    def fit(self) -> None:
        """
        Обучает модель RandomForestClassifier на тренировочных данных. 
        """
        self.X, self.y = self.preprocessor(self.train_data, fit_vectorizer=True)

        self.model.fit(self.X, self.y)

    def predict(self, path: str) -> np.ndarray:
        """
        Выполняет предсказания на данных, загруженных из указанного файла.
        
        :param path: путь к файлу с данными для предсказания.
        :return: массив с результатами предсказаний.
        """
        if not os.path.exists(self.model_pickle_path):
            raise ValueError('Модель не обучена')
        if not os.path.exists(self.preprocessor_pickle_path):
            raise ValueError('Препроцессор не обучен')

        self.X, self.y = self.preprocessor(path)

        return self.model.predict(self.X)
    
    def __calculate_metrics(self, predictions: np.ndarray) -> list[str]:
        """
        Рассчитывает метрики качества модели на основе предсказаний и истинных значений.
        
        :param predictions: массив с результатами предсказаний модели.
        :return: список строк с информацией о метриках.
        """
        tn, fp, fn, tp = confusion_matrix(self.y, predictions).ravel()

        metrics_info = [
            f"True positive: {tp}",
            f"False positive: {fp}",
            f"False negative: {fn}",
            f"True negative: {tn}",
            f"Accuracy: {accuracy_score(self.y, predictions):.4f}",
            f"Precision: {precision_score(self.y, predictions):.4f}",
            f"Recall: {recall_score(self.y, predictions):.4f}",
            f"F1: {f1_score(self.y, predictions):.4f}"
            ]

        return metrics_info
    
    def write_val_results(self) -> None:
        """
        Выполняет предсказания на валидационном наборе данных и записывает результаты в файл.        
        """
        predictions = self.predict(self.val_data)

        metrics_info = self.__calculate_metrics(predictions)

        with open(self.validation_results_path, 'w') as f:
            f.write("\n".join(metrics_info))

    def __get_feature_importance_explanation(self, i: int) -> str:
        """
        Генерирует объяснение предсказания для заданного примера, основываясь на важности top-k признаков.

        :param i: индекс примера в наборе данных.
        :return: строка с описанием важных признаков.
        """
        k = self.top_k[i]
        top_features = np.argsort(self.feature_importance[i])[-k:]
        important_libs = self.feature_names[top_features]
        explanation = ', '.join(important_libs)
        return f"File {i} predicted as virus due to libraries: {explanation}"
    
    def __create_feature_importance_array(self) -> None:
        """
        Создает массив "важностей" признаков из тестовой выборки. 
        """
        importances = self.model.feature_importances_
            
        self.file_features = self.X.toarray()
        self.feature_importance = importances * self.file_features

    def __create_top_k_array(self, k: int) -> None:
        """
        Создает массив top_k, где i-ый элемент массива - 
        min(k, количество признаков (библиотек) в i-ом примере из тестовой выборки).

        :param k: количество признаков для описания.
        """
        feature_count = np.sum(self.file_features, axis=-1)
        self.top_k = np.where(feature_count > k, k, feature_count)

    def write_pred_results(self, k: int = 3) -> None:
        """
        Выполняет предсказания на тестовом наборе данных и записывает результаты и объяснения в файлы.
        В объяснение включаются top-k признаков по важности с точки зрения модели.
        Если k > количества признаков в i-ом примере тестовой выборки, то в объяснение берутся все признаки. 

        :param k: количество признаков для описания.
        """
        predictions = self.predict(self.test_data)
        self.feature_names = self.preprocessor.get_feature_names_out()

        self.__create_feature_importance_array()
        self.__create_top_k_array(k)

        with open(self.prediction_results_path, 'w') as pred_file:
            pred_file.write("prediction\n" + "\n".join(np.vectorize(str)(predictions)))

        with open(self.explanaition_path, 'w') as expl_file:
            for i, pred in enumerate(predictions):
                explanation_file_string = pred * self.__get_feature_importance_explanation(i)   
                expl_file.write((i != 0) * "\n" + explanation_file_string)

    def dump(self) -> None:
        """
        Сохраняет текущее состояние модели и препроцессора в указанные файлы.
        """
        joblib.dump(self.model, self.model_pickle_path)
        joblib.dump(self.preprocessor, self.preprocessor_pickle_path)