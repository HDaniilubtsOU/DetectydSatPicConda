from osgeo import gdal
from osgeo import ogr, osr, gdal_array, gdalconst
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from skimage import exposure
import textwrap


# нужно написать функции для 3 видов спутников:
# 1) Спутники наблюдения за Землей (Landsat: 9 каналов) 2) Метеорологические спутники (NOAA AVHRR: 5 кан.)
# 3) Гиперспектральные спутники (Hyperion: до 220 спектральных каналов в узких диапазонах,
# охватывающих видимый и инфракрасный спектр)
# ___________________________________________________normalize_________________________________________________________
# imagery_ds = gdal.Open('data/multiband_imagery.tif', gdal.GA_ReadOnly)
# # print ('Bands count: %s' % imagery_ds.RasterCount)
#
# # Получаем количество каналов
# bands = imagery_ds.RasterCount
# print(f'Количество каналов: {bands}')
#
#
# channels = []
# # Получаем описание каждого канала
# for i in range(1, bands + 1):
#     # Предполагаем, что у вас 4 канала
#     band = imagery_ds.GetRasterBand(i)
#     channels.append(band.ReadAsArray())
#
#     # data = band.ReadAsArray()
#     # plt.imshow(data, cmap='gray')
#     # plt.title(f'Канал {i}')
#     # plt.colorbar()
#     # plt.show()
#
# # Например, RGB и NIR:
# red, green, blue, nir = channels
# # print(f"Минимум и максимум в красном канале: {red.min()}, {red.max()}")
# # print(f"Минимум и максимум в зеленом канале: {green.min()}, {green.max()}")
# # print(f"Минимум и максимум в голубом канале: {blue.min()}, {blue.max()}")
# # print(f"Минимум и максимум в нир канале: {nir.min()}, {nir.max()}")
#
# # Применение контрастного растяжения
# red_stretched = exposure.equalize_hist(red)
# green_stretched = exposure.equalize_hist(green)
# blue_stretched = exposure.equalize_hist(blue)
# nir_stretched = exposure.equalize_hist(nir)
#
# # нормализация значений канала относительно глобального максимума
# red_norm = np.clip(red / red.max(), 0, 1)
# green_norm = np.clip(green / green.max(), 0, 1)
# blue_norm = np.clip(blue / blue.max(), 0, 1)
# nir_norm = np.clip(nir / nir.max(), 0, 1)
# # print(f"Normalize для красного канала: {red_norm.min()}, {red_norm.max()}")
# # print(f"Normalize для зеленого канала: {green_norm.min()}, {green_norm.max()}")
# # print(f"Normalize для голубого канала: {blue_norm.min()}, {blue_norm.max()}")
# # print(f"Normalize для нир канала: {nir_norm.min()}, {nir_norm.max()}")
#
# # Применение контрастного растяжения
# red_stretched_norm = np.clip(red_stretched / red_stretched.max(), 0, 1)
# green_stretched_norm = np.clip(green_stretched / green_stretched.max(), 0, 1)
# blue_stretched_norm = np.clip(blue_stretched / blue_stretched.max(), 0, 1)
# nir_stretched_norm = np.clip(nir_stretched / nir_stretched.max(), 0, 1)
#
# # plt.imshow(red, cmap='gray')
# # plt.title(f'Канал Red')
# # plt.colorbar()
# # plt.show()
#

# # описываем функцию, которая будет нормализовать значения канала в диапазон от 0 до 1 (линейная нормализация используется для более естественного результата)
# def normalize(input_band):
#     min_value, max_value = input_band.min() * 1.0, input_band.max() * 1.0
#     return (input_band * 1.0 - min_value * 1.0) / (max_value * 1.0 - min_value)
# # описываем функцию, которая будет нормализовать значения канала с удалением выбросов
# def normalize2(input_band, clip_percentile=2):
#     lower, upper = np.percentile(input_band, (clip_percentile, 100 - clip_percentile))
#     clipped_band = np.clip(input_band, lower, upper)
#     return (clipped_band - lower) / (upper - lower)
# # Эта ^^^ версия обрезает выбросы (например, верхние и нижние 2% значений), что помогает сгладить сильные отклонения.
#
#
#
# imagery_ds_data = imagery_ds.ReadAsArray()
# # print (imagery_ds_data.shape)
#
# # собираем матрицу из нормализованных каналов
# rgb_normalized = np.dstack([normalize(imagery_ds_data[2]),
#                             normalize(imagery_ds_data[1]),
#                             normalize(imagery_ds_data[0])])
# # plt.imshow(rgb_normalized)
# # plt.title("RGB 0-1")
# # plt.axis('off')
# # plt.show()
#
# rgb_normalized_nir = np.dstack([normalize(imagery_ds_data[3])])
#
# # собираем матрицу из нормализованных каналов
# rgb_normalized2 = np.dstack([normalize2(imagery_ds_data[2]),
#                             normalize2(imagery_ds_data[1]),
#                             normalize2(imagery_ds_data[0])])
# # plt.imshow(rgb_normalized2)
# # plt.title("RGB с удалением выбросов")
# # plt.axis('off')
# # plt.show()
#
# rgb_normalized_nir2 = np.dstack([normalize2(imagery_ds_data[3])])
#
# rgb_image_normalized = np.dstack((red_norm, green_norm, blue_norm))
# # plt.imshow(rgb_image)
# # plt.title("RGB Глобальный максимум")
# # plt.axis('off')
# # plt.show()
#
# # # Закрываем набор данных
# # imagery_ds = None
# ___________________________________________________normalize_________________________________________________________

# Функция для открытия изображения
def open_multiband_image(file_path):
    """
    Открывает многоканальное изображение с помощью GDAL.

    :param file_path: Путь к файлу изображения
    :return: объект GDAL и количество каналов
    """
    dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Не удалось открыть файл {file_path}")

    bands = dataset.RasterCount
    print(f"Количество каналов: {bands}")
    return dataset, bands


# Функция для извлечения данных каналов
def get_channels(dataset, bands):
    """
    Извлекает данные всех каналов из изображения.

    :param dataset: GDAL-объект изображения
    :param bands: Количество каналов
    :return: список массивов каналов
    """
    channels = []
    for i in range(1, bands + 1):
        band = dataset.GetRasterBand(i)
        channels.append(band.ReadAsArray())
    return channels


def assign_channels(channels, band_names=None):
    """
    Распределяет каналы изображения и создаёт переменные для дальнейшей работы.

    :param channels: Список массивов данных каналов.
    :param band_names: Список названий каналов. Должен быть равен числу каналов или None.
                       Если None, каналы называются "Band1", "Band2", и т.д.
    :return: Словарь с распределёнными каналами.
    """
    if band_names is None:
        band_names = [f"Band{i + 1}" for i in range(len(channels))]

    # Проверка, что количество названий соответствует количеству каналов
    if len(band_names) < len(channels):
        raise ValueError("Недостаточно названий каналов для количества данных.")
    band_names = band_names[:len(channels)]  # Ограничиваем до доступных данных

    # Создание словаря с распределением
    channel_dict = {name: channel for name, channel in zip(band_names, channels)}

    # Создание переменных в глобальной области видимости
    globals().update(channel_dict)

    return channel_dict


# Функция для нормализации значений канала относительно глобального максимума
def normalize_band_global_max(input_band):
    """
    Нормализует значения канала в диапазон от 0 до 1.

    :param input_band: Массив данных канала
    :return: Нормализованный массив
    """
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


# Функция для контрастного растяжения
def stretch_contrast(input_band):
    """
    Применяет контрастное растяжение к каналу с помощью гистограммы.

    :param input_band: Массив данных канала
    :return: Массив с растянутым контрастом
    """
    return exposure.equalize_hist(input_band)


# Функция для визуализации одного канала
def plot_band(band, title, cmap='gray'):
    """
    Визуализирует один канал.

    :param band: Массив данных канала
    :param title: Заголовок графика
    :param cmap: Цветовая карта
    """
    plt.imshow(band, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.show()


# Функция для визуализации RGB-изображения
def plot_rgb(red, green, blue, title="RGB Composite"):
    """
    Визуализирует RGB-изображение.

    :param red: Красный канал
    :param green: Зеленый канал
    :param blue: Синий канал
    :param title: Заголовок графика
    """
    rgb_image = np.dstack((red, green, blue))
    plt.imshow(rgb_image)
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_quantity_three_rgb(red1, green1, blue1,
                        red2, green2, blue2,
                        red3, green3, blue3,
                        titles=("Image 1", "Image 2", "Image 3")):
    """
    Визуализирует три RGB-изображения в одном окне.

    :param red1, green1, blue1: Каналы для первого изображения
    :param red2, green2, blue2: Каналы для второго изображения
    :param red3, green3, blue3: Каналы для третьего изображения
    :param titles: Кортеж заголовков для каждого изображения
    """
    # Создание трёх RGB-изображений
    rgb_image1 = np.dstack((red1, green1, blue1))
    rgb_image2 = np.dstack((red2, green2, blue2))
    rgb_image3 = np.dstack((red3, green3, blue3))

    # Создание фигуры и трёх подграфиков
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 строка, 3 столбца

    # Первое изображение
    axs[0].imshow(rgb_image1)
    axs[0].set_title(titles[0])
    axs[0].axis('off')

    # Второе изображение
    axs[1].imshow(rgb_image2)
    axs[1].set_title(titles[1])
    axs[1].axis('off')

    # Третье изображение
    axs[2].imshow(rgb_image3)
    axs[2].set_title(titles[2])
    axs[2].axis('off')

    # Отображение графиков
    plt.tight_layout()
    plt.show()


def plot_histograms_three(red1, green1, blue1,
                          red2, green2, blue2,
                          red3, green3, blue3,
                          titles=("Image 1", "Image 2", "Image 3")):
    """
    Визуализирует гистограммы трёх RGB-изображений в одном окне.

    :param red1, green1, blue1: Каналы для первого изображения
    :param red2, green2, blue2: Каналы для второго изображения
    :param red3, green3, blue3: Каналы для третьего изображения
    :param titles: Кортеж заголовков для каждого изображения
    """
    fig_h, axs_h = plt.subplots(3, 1, figsize=(10, 15))  # 3 строки, 1 столбец

    # Вспомогательная функция для построения гистограммы одного изображения
    def plot_histogram(ax_h, red, green, blue, title):
        ax_h.hist(red.ravel(), bins=256, color='red', alpha=0.5, label='Red')
        ax_h.hist(green.ravel(), bins=256, color='green', alpha=0.5, label='Green')
        ax_h.hist(blue.ravel(), bins=256, color='blue', alpha=0.5, label='Blue')
        ax_h.set_title(title)
        ax_h.set_xlabel('Intensity')
        ax_h.set_ylabel('Frequency')
        ax_h.legend()

    # Гистограммы для первого изображения
    plot_histogram(axs_h[0], red1, green1, blue1, titles[0])

    # Гистограммы для второго изображения
    plot_histogram(axs_h[1], red2, green2, blue2, titles[1])

    # Гистограммы для третьего изображения
    plot_histogram(axs_h[2], red3, green3, blue3, titles[2])

    # Автоматическое выравнивание
    plt.tight_layout()
    plt.show()




# Основной скрипт
if __name__ == "__main__":
    file_path = 'data/multiband_imagery.tif'
    imagery_ds, num_bands = open_multiband_image(file_path)

    # Извлечение каналов
    channels = get_channels(imagery_ds, num_bands)
    # red, green, blue, nir = channels

    # Распределение каналов
    band_names = [
        "Blue", "Green", "Red", "NIR",
        "SWIR1", "SWIR2", "SWIR3", "PAN", "TIR"
    ]

    # Распределение и создание переменных
    channel_dict = assign_channels(channels, band_names[:num_bands])  # Учитываем только доступные каналы

    # Развёртывание в локальные переменные с безопасной проверкой
    Blue = channel_dict.get("Blue", None)
    Green = channel_dict.get("Green", None)
    Red = channel_dict.get("Red", None)
    NIR = channel_dict.get("NIR", None)
    SWIR1 = channel_dict.get("SWIR1", None)
    SWIR2 = channel_dict.get("SWIR2", None)
    SWIR3 = channel_dict.get("SWIR3", None)
    PAN = channel_dict.get("PAN", None)
    TIR = channel_dict.get("TIR", None)

    # # Теперь можно использовать переменные напрямую
    # print(f"Red Channel Min: {Red.min()}, Max: {Red.max()}")
    # print(f"Blue Channel Min: {Blue.min()}, Max: {Blue.max()}")
    # print(f"NIR Channel Min: {NIR.min()}, Max: {NIR.max()}")

    for channel_name in band_names:
        if channel_dict.get(channel_name) is None:
            print(f"Channel {channel_name} is not available.")
    print("\n")

    # Информация о каналах
    print("Информация о каналах:")
    for name, channel in channel_dict.items():
        print(f"{name} - Min: {channel.min()}, Max: {channel.max()}")
    print("\n")

    # Информация о каналах normalize_band_global_max
    print("Информация о каналах normalize_band_global_max:")
    for name, channel in channel_dict.items():
        print(f"{name} - "
              f"Min: {normalize_band_global_max(channel.min())}, "
              f"Max: {normalize_band_global_max(channel.max())}"
              )
    print("\n")

    # Информация о каналах линейная нормализация от 0 до 1
    print("Информация о каналах line_normalize:")
    for name, channel in channel_dict.items():
        print(
            f"{name} - "
            f"Min: {line_normalize(channel.min())}, "
            f"Max: {line_normalize(channel.max())}"
              )
    print("\n")

    # Информация о каналах normalize_with_delite_emissions
    print("Информация о каналах normalize_with_delite_emissions:")
    for name, channel in channel_dict.items():
        print(f"{name} - "
              f"Min: {normalize_with_delite_emissions(channel.min())}, "
              f"Max: {normalize_with_delite_emissions(channel.max())}"
              )
    print("\n")

    # Примеры обработки
    # if "Red" in named_channels and "Green" in named_channels and "Blue" in named_channels:
    if Red is not None and Green is not None and Blue is not None:
        plot_rgb(
            normalize_band_global_max(Red),
            normalize_band_global_max(Green),
            normalize_band_global_max(Blue),
        )

    if Red is not None and Green is not None and Blue is not None:
        plot_rgb(
            line_normalize(Red),
            line_normalize(Green),
            line_normalize(Blue),
        )

    if Red is not None and Green is not None and Blue is not None:
        plot_rgb(
            normalize_with_delite_emissions(Red),
            normalize_with_delite_emissions(Green),
            normalize_with_delite_emissions(Blue),
        )

    if Red is not None and Green is not None and Blue is not None:
        red1, green1, blue1 = \
            normalize_band_global_max(Red), \
            normalize_band_global_max(Green),\
            normalize_band_global_max(Blue)
        red2, green2, blue2 = \
            line_normalize(Red), \
            line_normalize(Green), \
            line_normalize(Blue)
        red3, green3, blue3 = \
            normalize_with_delite_emissions(Red), \
            normalize_with_delite_emissions(Green), \
            normalize_with_delite_emissions(Blue)

        # Вызов функции для визуализации
        plot_quantity_three_rgb(
            red1, green1, blue1,
            red2, green2, blue2,
            red3, green3, blue3,
            titles=("Default Normalize (global_max)", "Line Normalize", "Clip Normalize")
        )

        # Вызов функции для отображения гистограмм
        plot_histograms_three(
            red1, green1, blue1,
            red2, green2, blue2,
            red3, green3, blue3,
            titles=("Default Normalize Histogram", "Line Normalize Histogram", "Clip Normalize Histogram")
        )

    if Red is not None and Green is not None and Blue is not None:
        # Предположим, у вас есть три набора нормализованных RGB-каналов
        red1, green1, blue1 = \
            stretch_contrast(normalize_band_global_max(Red)),\
            stretch_contrast(normalize_band_global_max(Green)),\
            stretch_contrast(normalize_band_global_max(Blue))
        red2, green2, blue2 = \
            stretch_contrast(line_normalize(Red)), \
            stretch_contrast(line_normalize(Green)), \
            stretch_contrast(line_normalize(Blue))
        red3, green3, blue3 = \
            stretch_contrast(normalize_with_delite_emissions(Red)), \
            stretch_contrast(normalize_with_delite_emissions(Green)), \
            stretch_contrast(normalize_with_delite_emissions(Blue))

        # Вызов функции для визуализации
        plot_quantity_three_rgb(
            red1, green1, blue1,
            red2, green2, blue2,
            red3, green3, blue3,
            titles=("Default Normalize (global_max)", "Line Normalize", "Clip Normalize")
        )

        # Вызов функции для отображения гистограмм
        plot_histograms_three(
            red1, green1, blue1,
            red2, green2, blue2,
            red3, green3, blue3,
            titles=("Default Normalize Histogram", "Line Normalize Histogram", "Clip Normalize Histogram")
        )

    if NIR is not None:
        plot_band(normalize_band_global_max(NIR), "Ближний инфракрасный (NIR) normalize_band_global_max")

    if NIR is not None:
        plot_band(line_normalize(NIR), "Ближний инфракрасный (NIR) line_normalize")

    if NIR is not None:
        plot_band(normalize_with_delite_emissions(NIR), "Ближний инфракрасный (NIR) normalize_with_delite_emissions")








    """добавить работу с другими каналами"""

    for swir_name in [SWIR1, SWIR2, SWIR3]:                                                       #????????????????????
        if swir_name is not None:
            plot_band(normalize_band_global_max(swir_name), f"{swir_name} (Средний инфракрасный)")

    if PAN is not None:
        plot_band(normalize_band_global_max(PAN), "Панхроматический канал (PAN)")

    if TIR is not None:
        plot_band(normalize_band_global_max(TIR), "Тепловое инфракрасное излучение (TIR)")



    # # Нормализация
    # red_norm = normalize_band_global_max(red)
    # green_norm = normalize_band_global_max(green)
    # blue_norm = normalize_band_global_max(blue)
    # nir_norm = normalize_band_global_max(nir)

    # # Применение контрастного растяжения
    # red_stretched = stretch_contrast(red)
    # green_stretched = stretch_contrast(green)
    # blue_stretched = stretch_contrast(blue)
    # nir_stretched = stretch_contrast(nir)

    # # Нормализация растянутых каналов
    # red_stretched_norm = normalize_band_global_max(red_stretched)
    # green_stretched_norm = normalize_band_global_max(green_stretched)
    # blue_stretched_norm = normalize_band_global_max(blue_stretched)
    # nir_stretched_norm = normalize_band_global_max(nir_stretched)

    # # Визуализация каналов
    # plot_band(red, "Красный канал")
    # plot_band(green, "Зеленый канал")
    # plot_band(blue, "Синий канал")
    # plot_band(nir, "NIR канал")

    # # Визуализация RGB с нормализацией
    # plot_rgb(red_norm, green_norm, blue_norm, "RGB (нормализованное)")
    #
    # # Визуализация RGB с растянутым контрастом
    # plot_rgb(red_stretched_norm, green_stretched_norm, blue_stretched_norm, "RGB (растянутый контраст)")

