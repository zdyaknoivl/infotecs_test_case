import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class DataPreprocessor:
    def __init__(self) -> None:
        """
        Инициализирует объект класса DataPreprocessor.
        """
        self.vectorizer = CountVectorizer(tokenizer=self.tokenize) 
    
    def tokenize(self, x: str) -> list[str]:
        """
        Разбивает строку на токены (библиотеки), используя запятую в качестве разделителя.

        :param x: строка для токенизации.
        :return: список токенов.
        """
        return x.split(',')
    
    def split_data(self, path: str, fit_vectorizer: bool = False) -> tuple:
        """
        Разделяет данные на признаки и целевую переменную, при необходимости обучая векторизатор.

        :param path: путь к файлу с данными.
        :param fit_vectorizer: флаг, указывающий, следует ли обучать векторизатор.
        :return: кортеж (X, y), где X - признаки (библиотеки), y - целевая переменная (является ли файл зловредным).
        """
        data = pd.read_csv(path, sep='\t')
        X = (
            self.vectorizer.fit_transform(data['libs']) 
            if fit_vectorizer else self.vectorizer.transform(data['libs'])
        )
        y = None if 'is_virus' not in data.columns else data['is_virus']

        return X, y

    def get_feature_names_out(self) -> list[str]:
        """
        Возвращает имена признаков (библиотек) после векторизации.

        :return: список имен признаков.
        """
        return self.vectorizer.get_feature_names_out()

    def __call__(self, path: str, fit_vectorizer: bool = False) -> tuple:
        """
        Вызывает метод split_data с указанными параметрами.

        :param path: путь к файлу с данными.
        :param fit_vectorizer: флаг, указывающий, следует ли обучать векторизатор.
        :return: результат выполнения метода split_data.
        """
        return self.split_data(path, fit_vectorizer)