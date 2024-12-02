"""Normalized Difference Vegetation Index"""
import numpy as np


# Применяется для оценки состояния растительности. Значения близкие к 1 указывают на плотный растительный покров.
def calculate_ndvi(nir_channel, red_channel, epsilon=1e-10):
    """
    Вычисляет индекс NDVI (Normalized Difference Vegetation Index).

    :param nir_channel: Массив данных ближнего инфракрасного канала (NIR)
    :param red_channel: Массив данных красного канала (Red)
    :param epsilon: Небольшое число для предотвращения деления на ноль
    :return: Массив NDVI с диапазоном значений от -1 до 1
    """
    # Преобразуем данные в формат с плавающей точкой для точности вычислений
    nir = nir_channel.astype(float)    # float16 / 32
    red = red_channel.astype(float)

    # # Вычисляем NDVI с защитой от деления на ноль
    # ndvi = (nir - red) / (nir + red + epsilon)

    # Игнорируем предупреждения о делении на ноль
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / (nir + red + epsilon)

        # Заменяем NaN (вызванные делением 0/0) на 0
        ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=1.0, neginf=-1.0)

    # Обработка деления на ноль (замена NaN на 0)
    # ndvi = np.nan_to_num(ndvi)
    # ndvi[np.isnan(ndvi)] = 0  # Убираем NaN значения

    return ndvi


def classify_ndvi(ndvi):
    """
    Классифицирует NDVI по категориям.

    :param ndvi: Массив NDVI значений
    :return: Массив категорий NDVI
    """
    # classification = np.zeros_like(ndvi, dtype=np.uint8)
    classification_ndvi = np.zeros_like(ndvi)

    # Условия классификации
    classification_ndvi[(ndvi >= -1) & (ndvi < 0)] = 1  # Вода
    classification_ndvi[(ndvi >= 0) & (ndvi < 0.2)] = 2  # Слабая растительность
    classification_ndvi[(ndvi >= 0.2) & (ndvi < 0.5)] = 3  # Умеренная растительность
    classification_ndvi[(ndvi >= 0.5)] = 4  # Плотная растительность

    return classification_ndvi