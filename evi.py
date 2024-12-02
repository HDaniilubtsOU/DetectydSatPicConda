"""Enhanced Vegetation Index"""
import numpy as np


# Усиленный вегетационный индекс (EVI) используется для улучшения оценки плотности растительности,
# устраняя атмосферные эффекты и влияние почвы.
# def calculate_evi(nir, red, blue, G=2.5, C1=6.0, C2=7.5, L=1.0):
def calculate_evi(nir, red, blue, G=2.5, C1=6.0, C2=7.5, L=1.0):
    """
    Рассчитывает индекс EVI.

    :param nir: Ближний инфракрасный канал.
    :param red: Красный канал.
    :param blue: Синий канал.
    :param G: Коэффициент усиления.
    :param C1: Коэффициент для коррекции атмосферы (Red).
    :param C2: Коэффициент для коррекции атмосферы (Blue).
    :param L: Поправочный коэффициент.
    :return: EVI массив.
    """
    nir = nir.astype(float)
    red = red.astype(float)
    blue = blue.astype(float)
    denominator = (nir + C1 * red - C2 * blue + L)
    evi = G * (nir - red) / denominator

    # Обработка NaN и бесконечных значений
    evi[np.isnan(evi)] = 0
    evi[np.isinf(evi)] = 0
    return evi


# def classify_evi(evi):
#     """
#     Классифицирует EVI по категориям.
#
#     :param ndvi: Массив EVI значений
#     :return: Массив категорий EVI
#     """
#     # classification = np.zeros_like(ndvi, dtype=np.uint8)
#     classification_evi = np.zeros_like(evi)
#
#     # Условия классификации
#     classification_evi[(evi >= -1) & (evi < 0)] = 1  # Вода
#     classification_evi[(evi >= 0) & (evi < 0.2)] = 2  # Слабая растительность
#     classification_evi[(evi >= 0.2) & (evi < 0.6)] = 3  # Умеренная растительность
#     classification_evi[(evi >= 0.6)] = 4  # Плотная растительность
#
#     return classification_evi

# EVI Classification
def classify_evi(evi):
    """
    Классифицирует EVI по категориям.

    :param evi: Массив EVI значений.
    :return: Массив категорий EVI.
    """
    classification_evi = np.zeros_like(evi)
    classification_evi[(evi < 0)] = 1  # Low
    classification_evi[(evi >= 0) & (evi < 0.2)] = 2  # Moderate
    classification_evi[(evi >= 0.2) & (evi < 0.5)] = 3  # High
    classification_evi[(evi >= 0.5) & (evi < 1)] = 4  # Very High
    classification_evi[(evi >= 1)] = 5  # Extreme
    return classification_evi