"""Normalized Difference Water Index"""
import numpy as np


def calculate_ndwi(nir_channel, green_channel, epsilon=1e-10):
    """
    Вычисляет индекс NDWI (Normalized Difference Water Index).

    :param nir_channel: Массив данных ближнего инфракрасного канала (NIR)
    :param green_channel: Массив данных зелёного канала (Green)
    :param epsilon: Небольшое число для предотвращения деления на ноль
    :return: Массив NDWI с диапазоном значений от -1 до 1
    """
    nir = nir_channel.astype(float)
    green = green_channel.astype(float)

    ndwi = (green - nir) / (green + nir + epsilon)
    return ndwi


# NDWI Classification
def classify_ndwi(ndwi):
    """
    Классифицирует NDWI по категориям.

    :param ndwi: Массив NDWI значений.
    :return: Массив категорий NDWI.
    """
    classification_ndwi = np.zeros_like(ndwi)
    classification_ndwi[(ndwi < 0)] = 1  # Low Water Content
    classification_ndwi[(ndwi >= 0) & (ndwi < 0.3)] = 2  # Moderate Water Content
    classification_ndwi[(ndwi >= 0.3)] = 3  # High Water Content
    return classification_ndwi