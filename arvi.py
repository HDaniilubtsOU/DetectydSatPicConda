"""Atmospherically Resistant Vegetation Index"""
import numpy as np


def calculate_arvi(nir_channel, red_channel, blue_channel, epsilon=1e-10):
    """
    Вычисляет индекс ARVI (Atmospherically Resistant Vegetation Index).

    :param nir_channel: Массив данных ближнего инфракрасного канала (NIR)
    :param red_channel: Массив данных красного канала (Red)
    :param blue_channel: Массив данных синего канала (Blue)
    :param epsilon: Небольшое число для предотвращения деления на ноль
    :return: Массив ARVI с диапазоном значений от -1 до 1
    """
    nir = nir_channel.astype(float)
    red = red_channel.astype(float)
    blue = blue_channel.astype(float)

    arvi = (nir - (2 * red - blue)) / (nir + (2 * red - blue) + epsilon)
    return arvi


# ARVI Classification
def classify_arvi(arvi):
    """
    Классифицирует ARVI по категориям.

    :param arvi: Массив ARVI значений.
    :return: Массив категорий ARVI.
    """
    classification_arvi = np.zeros_like(arvi)
    classification_arvi[(arvi >= -1) & (arvi < 0)] = 1  # Low Vegetation
    classification_arvi[(arvi >= 0) & (arvi < 0.2)] = 2  # Moderate Vegetation
    classification_arvi[(arvi >= 0.2) & (arvi < 0.5)] = 3  # High Vegetation
    classification_arvi[(arvi >= 0.5)] = 4  # Very High Vegetation
    return classification_arvi