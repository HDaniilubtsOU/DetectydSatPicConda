"""Water Ratio Index"""
import numpy as np


def calculate_wri(green_channel, red_channel, nir_channel, swir_channel, epsilon=1e-10):
    """
    Вычисляет индекс WRI (Water Ratio Index).

    :param green_channel: Массив данных зелёного канала (Green)
    :param red_channel: Массив данных красного канала (Red)
    :param nir_channel: Массив данных ближнего инфракрасного канала (NIR)
    :param swir_channel: Массив данных среднего инфракрасного канала (SWIR)
    :param epsilon: Небольшое число для предотвращения деления на ноль
    :return: Массив WRI
    """
    green = green_channel.astype(float)
    red = red_channel.astype(float)
    nir = nir_channel.astype(float)
    swir = swir_channel.astype(float)

    wri = (green + red) / (nir + swir + epsilon)
    return wri


# WRI Classification
def classify_wri(wri):
    """
    Классифицирует WRI по категориям.

    :param wri: Массив WRI значений.
    :return: Массив категорий WRI.
    """
    classification_wri = np.zeros_like(wri)
    classification_wri[(wri < 1)] = 1  # Low Water Presence
    classification_wri[(wri >= 1) & (wri < 2)] = 2  # Moderate Water Presence
    classification_wri[(wri >= 2)] = 3  # High Water Presence
    return classification_wri