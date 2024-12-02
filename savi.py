"""Soil-Adjusted Vegetation Index"""
import numpy as np


def calculate_savi(nir_channel, red_channel, L=0.5, epsilon=1e-10):
    """
    Вычисляет индекс SAVI (Soil-Adjusted Vegetation Index).

    :param nir_channel: Массив данных ближнего инфракрасного канала (NIR)
    :param red_channel: Массив данных красного канала (Red)
    :param L: Коэффициент поправки на почву (по умолчанию 0.5)
    :param epsilon: Небольшое число для предотвращения деления на ноль
    :return: Массив SAVI с диапазоном значений от -1 до 1
    """
    nir = nir_channel.astype(float)
    red = red_channel.astype(float)

    savi = ((nir - red) / (nir + red + L + epsilon)) * (1 + L)
    return savi


# SAVI Classification
def classify_savi(savi):
    """
    Классифицирует SAVI по категориям.

    :param savi: Массив SAVI значений.
    :return: Массив категорий SAVI.
    """
    classification_savi = np.zeros_like(savi)
    classification_savi[(savi < 0.1)] = 1  # Low
    classification_savi[(savi >= 0.1) & (savi < 0.3)] = 2  # Moderate
    classification_savi[(savi >= 0.3) & (savi < 0.5)] = 3  # High
    classification_savi[(savi >= 0.5)] = 4  # Very High
    return classification_savi