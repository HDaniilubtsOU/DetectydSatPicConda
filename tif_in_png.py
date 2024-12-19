import numpy as np
import os
from PIL import Image


def convert_normalized_array_to_png(normalized_array, output_file):
    """
    Преобразует многоканальный нормализованный массив (RGB) в изображение формата PNG.

    :param normalized_array: Нормализованный трёхканальный массив (значения от 0 до 1)
    :param output_file: Путь к выходному .png файлу
    :return: Путь к сохранённому файлу .png
    """
    # Создаём директорию, если её нет
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Проверка на трёхмерность массива
    if normalized_array.ndim != 3 or normalized_array.shape[2] != 3:
        raise ValueError("Входной массив должен быть трёхмерным с тремя каналами (RGB).")

    # Преобразуем значения массива из диапазона [0, 1] в [0, 255]
    scaled_array = (normalized_array * 255).astype(np.uint8)

    # Создаём изображение с помощью Pillow
    image = Image.fromarray(scaled_array, mode="RGB or RGBA")

    # Сохраняем изображение в формате PNG
    image.save(output_file, format="PNG")

    print(f"Файл успешно сохранён как {output_file}")
    return output_file


def convert_rgb_to_rgba_and_save(normalized_rgb_array, output_file, alpha_value=1.0):
    """
    Преобразует нормализованный RGB массив в RGBA и сохраняет как PNG.

    :param normalized_rgb_array: Нормализованный RGB массив (значения от 0 до 1)
    :param output_file: Путь к выходному .png файлу
    :param alpha_value: Значение прозрачности (0 - полностью прозрачное, 1 - полностью непрозрачное)
                        Или массив такого же размера, как RGB, для заданного прозрачного канала.
    :return: Путь к сохранённому файлу .png
    """
    # Создаём директорию, если её нет
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Проверяем входные данные
    if normalized_rgb_array.ndim != 3 or normalized_rgb_array.shape[2] != 3:
        raise ValueError("Входной массив должен быть трёхмерным с тремя каналами (RGB).")

    # Создаём альфа-канал
    if isinstance(alpha_value, (int, float)):
        alpha_channel = np.full(
            (normalized_rgb_array.shape[0], normalized_rgb_array.shape[1]),
            alpha_value,
            dtype=np.float32,
        )
    elif isinstance(alpha_value, np.ndarray) and alpha_value.shape[:2] == normalized_rgb_array.shape[:2]:
        alpha_channel = alpha_value
    else:
        raise ValueError("Альфа-канал должен быть числом или массивом такого же размера, как RGB.")

    # Ограничиваем альфа-канал в пределах [0, 1]
    alpha_channel = np.clip(alpha_channel, 0, 1)

    # Добавляем альфа-канал к RGB, формируя RGBA
    rgba_array = np.dstack((normalized_rgb_array, alpha_channel))

    # Преобразуем в диапазон [0, 255]
    scaled_array = (rgba_array * 255).astype(np.uint8)

    # Сохраняем как PNG
    image = Image.fromarray(scaled_array, mode="RGBA")
    image.save(output_file, format="PNG")

    print(f"Файл успешно сохранён как {output_file}")
    return output_file


def convert_rgb_to_rgba_with_condition(normalized_rgb_array, output_file, transparency_condition=None):
    """
    Преобразует RGB массив в RGBA с условной прозрачностью и сохраняет как PNG.

    :param normalized_rgb_array: Нормализованный RGB массив (значения от 0 до 1)
    :param output_file: Путь к выходному .png файлу
    :param transparency_condition: Функция или условие для задания прозрачности. Должна возвращать True для прозрачных пикселей.
                                    Если None, то альфа-канал будет полностью непрозрачным.
    :return: Путь к сохранённому файлу .png
    """

    # Создаём директорию, если её нет
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Проверяем входные данные
    if normalized_rgb_array.ndim != 3 or normalized_rgb_array.shape[2] != 3:
        raise ValueError("Входной массив должен быть трёхмерным с тремя каналами (RGB).")

    # Создаём альфа-канал
    alpha_channel = np.ones((normalized_rgb_array.shape[0], normalized_rgb_array.shape[1]), dtype=np.float32)

    # Если задано условие, применяем его
    if transparency_condition is not None:
        transparent_mask = transparency_condition(normalized_rgb_array)
        alpha_channel[transparent_mask] = 0.0  # Прозрачность для удовлетворяющих условию

    # Ограничиваем альфа-канал в пределах [0, 1]
    alpha_channel = np.clip(alpha_channel, 0, 1)

    # Добавляем альфа-канал к RGB, формируя RGBA
    rgba_array = np.dstack((normalized_rgb_array, alpha_channel))

    # Преобразуем в диапазон [0, 255]
    scaled_array = (rgba_array * 255).astype(np.uint8)

    # Сохраняем как PNG
    image = Image.fromarray(scaled_array, mode="RGBA")
    image.save(output_file, format="PNG")

    print(f"Файл успешно сохранён как {output_file}")
    return output_file



