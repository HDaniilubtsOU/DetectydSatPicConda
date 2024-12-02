"""Modified Normalized Difference Water Index"""
import numpy as np


def calculate_mndwi(green_channel, swir_channel, epsilon=1e-10):
    """
    Вычисляет индекс MNDWI (Modified Normalized Difference Water Index).

    :param green_channel: Массив данных зелёного канала (Green)
    :param swir_channel: Массив данных среднего инфракрасного канала (SWIR)
    :param epsilon: Небольшое число для предотвращения деления на ноль
    :return: Массив MNDWI с диапазоном значений от -1 до 1
    """
    green = green_channel.astype(float)
    swir = swir_channel.astype(float)

    mndwi = (green - swir) / (green + swir + epsilon)
    return mndwi


# MNDWI Classification
def classify_mndwi(mndwi):
    """
    Классифицирует MNDWI по категориям.

    :param mndwi: Массив MNDWI значений.
    :return: Массив категорий MNDWI.
    """
    classification_mndwi = np.zeros_like(mndwi)
    classification_mndwi[(mndwi < 0)] = 1  # Low Water
    classification_mndwi[(mndwi >= 0) & (mndwi < 0.5)] = 2  # Moderate Water
    classification_mndwi[(mndwi >= 0.5)] = 3  # High Water
    return classification_mndwi