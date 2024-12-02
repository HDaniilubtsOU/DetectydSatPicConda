import numpy as np


# Функция для нормализации значений канала относительно глобального максимума
def normalize_band_global_max(input_band):
    """
    Нормализует значения канала в диапазон от 0 до 1.

    :param input_band: Массив данных канала
    :return: Нормализованный массив
    """
    max_value = input_band.max()
    if max_value == 0:
        return np.zeros_like(input_band)  # Возвращает массив из нулей, если максимум равен 0
    return np.clip(input_band / input_band.max(), 0, 1)


# описываем функцию, которая будет нормализовать значения канала в диапазон от 0 до 1
# (линейная нормализация используется для более естественного результата)
def line_normalize(input_band):
    """
        Линейная нормализация от 0 до 1.
        :param input_band: Массив данных канала
        :return: Нормализованный массив
        """
    min_value, max_value = input_band.min(), input_band.max()
    if max_value == min_value:  # Защита от деления на 0
        return np.zeros_like(input_band)
    return (input_band - min_value) / (max_value - min_value)


# описываем функцию, которая будет нормализовать значения канала с удалением выбросов
def normalize_with_delite_emissions(input_band, clip_percentile = 2):
    """
    :param input_band: Массив данных канала
    :param clip_percentile: обрезает (роцентиль = 2%) верхние и нижние  значений
    """
    lower, upper = np.percentile(input_band, (clip_percentile, 100 - clip_percentile))
    if upper == lower:  # Защита от деления на 0
        return np.zeros_like(input_band)
    clipped_band = np.clip(input_band, lower, upper)
    return (clipped_band - lower) / (upper - lower)
# Эта ^^^ версия обрезает выбросы (например, верхние и нижние 2% значений), что помогает сгладить сильные отклонения.