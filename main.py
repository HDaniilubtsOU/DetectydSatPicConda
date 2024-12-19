from osgeo import gdal
from osgeo import ogr, osr, gdal_array, gdalconst
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.image as mpimg
import numpy as np
from skimage_pac import exposure
import textwrap

import arvi
import cloud_processing
import evi
import mndwi
import ndvi
import ndwi
import savi
import wri

import types_normalize
import visualizations_plot
import visualizations_plot_histogram
import tif_in_png


# нужно написать функции для 3 видов спутников:
# 1) Спутники наблюдения за Землей (Landsat: 9 каналов)
# 2) Метеорологические спутники (NOAA AVHRR: 5 кан.)
# 3) Гиперспектральные спутники (Hyperion: до 220 спектральных каналов в узких диапазонах, охватывающих видимый и инфракрасный спектр)


# Cloud processing
# Регрессия и ее виды
# библиотеки skimage(старый деп) и sklearn(неиронки)


# заняться Атмосферной коррекцией
# Общие алгоритмы атмосферной коррекции:
# ATCOR2 (Atmospheric Correction for Flat Terrain 2)
# FLAASH (Fast Line-of-Sight Atmospheric Analysis of Spectral Hypercubes)
# DOS1 (Dark Object Subtraction 1)
# LaSRC (Land Surface Reflectance Code)
# iCOR (Image Correction for Atmospheric Effects)


# присутствуют индексы которые могут дать информацию исключительно из RGB

# GIS-based analysis as independent predictor variables


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


def extract_metadata(file_path):
    """
    Извлекает метаданные из многоканального изображения с помощью GDAL и разделяет их по переменным.

    :param file_path: Путь к файлу изображения
    :return: Словарь с метаданными
    """
    # Открываем файл
    dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Не удалось открыть файл {file_path}")

    # Извлекаем основные метаданные
    rows = dataset.RasterYSize
    cols = dataset.RasterXSize
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    # Разбиваем geo_transform на составляющие
    origin_x = geo_transform[0]  # Координата X верхнего левого угла
    pixel_width = geo_transform[1]  # Размер пикселя по оси X
    rotation_x = geo_transform[2]  # Поворот по оси X
    origin_y = geo_transform[3]  # Координата Y верхнего левого угла
    rotation_y = geo_transform[4]  # Поворот по оси Y
    pixel_height = geo_transform[5]  # Размер пикселя по оси Y (отрицательный)

    # Формируем словарь с результатами
    metadata = {
        "rows": rows,
        "cols": cols,
        "geo_transform": {
            "origin_x": origin_x,
            "pixel_width": pixel_width,
            "rotation_x": rotation_x,
            "origin_y": origin_y,
            "rotation_y": rotation_y,
            "pixel_height": pixel_height
        },
        "projection": projection
    }

    return metadata


def display_channel_array(channel_array, channel_name):
    """
    Отображает массив данных для указанного канала.

    :param channel_array: Массив данных канала
    :param channel_name: Имя канала
    """
    if channel_array is None:
        print(f"Канал {channel_name} отсутствует.")
    else:
        print(f"Массив данных для канала {channel_name}:")
        print(channel_array)


# def remove_black_zones_and_save_simple(dataset):
#     """
#     Удаляет черные зоны изображения, оставляя только полезную область,
#     используя битовую маску с помощью cv.bitwise_and.
#
#     :param dataset: GDAL Dataset — открытый объект GDAL
#     :return: GDAL Dataset — новый объект с обрезанным изображением
#     """
#     if dataset is None:
#         raise ValueError("Входной GDAL Dataset равен None.")
#
#     # Извлечение метаинформации
#     transform = dataset.GetGeoTransform()
#     projection = dataset.GetProjection()
#     bands = dataset.RasterCount
#
#     if bands == 0:
#         raise ValueError("GDAL Dataset не содержит каналов.")
#
#     # Читаем изображение в массив
#     arrays = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(bands)]
#     stacked_image = np.stack(arrays, axis=0)  # (bands, height, width)
#
#     # Создаем маску: любая область с ненулевым пикселем становится белой
#     mask = np.any(stacked_image > 0, axis=0).astype(np.uint8) * 255  # (height, width)
#
#     plt.figure(figsize=(6, 6))
#     plt.imshow(mask, cmap='gray')
#     plt.title("Маска по контурам")
#     plt.axis('off')
#     plt.show()
#
#     # # Применяем маску ко всем каналам изображения
#     # masked_bands = [cv.bitwise_and(band, band, mask=mask) for band in stacked_image]
#     # masked_image = np.stack(masked_bands, axis=0)  # (bands, height, width)
#     #
#     # # Находим ограничивающий прямоугольник
#     # contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     # if not contours:
#     #     raise ValueError("Все данные на изображении пустые или черные.")
#     #
#     # x, y, w, h = cv.boundingRect(np.vstack(contours))
#     #
#     # # Обрезаем изображение по найденным границам
#     # cropped_image = masked_image[:, y:y + h, x:x + w]
#     #
#     # mean_cropped_image = np.mean(cropped_image, axis=0)
#     # plt.figure(figsize=(8, 8))
#     # plt.imshow(mean_cropped_image, cmap='gray')
#     # plt.title("Результат применения маски")
#     # plt.axis('off')
#     # plt.show()
#     #
#     # # Обновляем геопривязку
#     # new_transform = (
#     #     transform[0] + x * transform[1],  # Новое начало по X
#     #     transform[1],
#     #     transform[2],
#     #     transform[3] + y * transform[5],  # Новое начало по Y
#     #     transform[4],
#     #     transform[5],
#     # )
#     #
#     # # Создаем временное изображение в памяти
#     # driver = gdal.GetDriverByName('MEM')
#     # output_ds = driver.Create(
#     #     '', w, h, bands, gdal_array.NumericTypeCodeToGDALTypeCode(cropped_image.dtype)
#     # )
#     # output_ds.SetGeoTransform(new_transform)
#     # output_ds.SetProjection(projection)
#     #
#     # # Записываем данные в объект GDAL
#     # for i in range(bands):
#     #     output_ds.GetRasterBand(i + 1).WriteArray(cropped_image[i])
#     #
#     # print("Черные зоны удалены. Обрезанное изображение готово для дальнейшей обработки.")
#     #
#     # return output_ds
#     # Находим контуры маски
#     contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         raise ValueError("Маска не содержит полезных данных.")
#
#     # Создаем объединённый контур маски
#     all_contours = np.vstack(contours)
#
#     # Обрезаем изображение строго по маске
#     x, y, w, h = cv.boundingRect(all_contours)  # Находим прямоугольник, в который вписывается маска
#     precise_mask = np.zeros_like(mask, dtype=np.uint8)
#     cv.drawContours(precise_mask, [all_contours], -1, color=255, thickness=cv.FILLED)
#
#     # Применяем маску к каждому каналу
#     masked_bands = [cv.bitwise_and(band, band, mask=precise_mask) for band in stacked_image]
#     masked_image = np.stack(masked_bands, axis=0)[:, y:y + h, x:x + w]  # Обрезаем по маске
#
#     # Обновляем геопривязку для обрезанного изображения
#     new_transform = (
#         transform[0] + x * transform[1],  # Новое начало по X
#         transform[1],
#         transform[2],
#         transform[3] + y * transform[5],  # Новое начало по Y
#         transform[4],
#         transform[5],
#     )
#
#     # Создаём временное изображение в памяти
#     driver = gdal.GetDriverByName('MEM')
#     output_ds = driver.Create(
#         '', masked_image.shape[2], masked_image.shape[1], bands,
#         gdal_array.NumericTypeCodeToGDALTypeCode(masked_image.dtype)
#     )
#     output_ds.SetGeoTransform(new_transform)
#     output_ds.SetProjection(projection)
#
#     # Записываем данные в новый GDAL Dataset
#     for i in range(bands):
#         output_ds.GetRasterBand(i + 1).WriteArray(masked_image[i])
#
#     print("Черный фон полностью удален. Изображение вырезано строго по маске.")
#
#     return output_ds
#     # # Нахождение непустых границ маски
#     # # Применяем маску к каждому каналу
#     # masked_bands = [cv.bitwise_and(band, band, mask=mask) for band in stacked_image]
#     # masked_image = np.stack(masked_bands, axis=0)  # (bands, height, width)
#     #
#     # # Создаём временное изображение в памяти
#     # driver = gdal.GetDriverByName('MEM')
#     # output_ds = driver.Create(
#     #     '', dataset.RasterXSize, dataset.RasterYSize, bands,
#     #     gdal_array.NumericTypeCodeToGDALTypeCode(masked_image.dtype)
#     # )
#     # output_ds.SetGeoTransform(transform)
#     # output_ds.SetProjection(projection)
#     #
#     # # Записываем данные в объект GDAL
#     # for i in range(bands):
#     #     output_ds.GetRasterBand(i + 1).WriteArray(masked_image[i])
#     #
#     # print("Маска применена, размеры изображения сохранены.")
#     #
#     # return output_ds

# нужно подумать про удаление за пределами маски, и оставляете только пиксели внутри самой крупной фигуры.
# Это приводит к тому, что размер массива становится (bands, N), где 𝑁 — количество пикселей внутри маски.
# Это одномерный массив, который нельзя преобразовать обратно в исходный двумерный размер (height, width).
# ЕСТЬ ЛИ ВОЗМОЖНОСТЬ ВЫКИНУТЬ ИЗ МАССИВА ФИГУРУ И ПРИ ЭТОМ СОХРАНИТЬ РАЗМЕРНОСТЬ МАССИВА
def remove_black_zones_and_save_simple(dataset):
    """
    Проверяет изображение на чёрные зоны, добавляет альфа-канал для прозрачности, обрезает полезную область.
    Возвращает данные изображения в массиве, а также обновлённые параметры геопривязки.

    :param dataset: GDAL Dataset — открытый объект GDAL
    :return: tuple:
        - image_with_alpha: numpy.ndarray — (bands + 1, height, width), данные изображения с альфа-каналом
        - transform: tuple — обновлённая геопривязка
        - projection: str — проекция изображения
    """
    # Извлечение данных
    bands = dataset.RasterCount
    transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    # Читаем каналы изображения в массив
    arrays = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(bands)]
    stacked_image = np.stack(arrays, axis=0)  # (bands, height, width)

    # Преобразуем изображение в формат OpenCV (height, width, bands)
    stacked_image_cv = np.moveaxis(stacked_image, 0, -1)  # (height, width, bands)

    # Создаем градации серого для генерации маски
    gray = np.mean(stacked_image_cv, axis=-1).astype(np.uint8)

    # Применение пороговой обработки для создания бинарной маски
    _, mask = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)

    # Инверсия маски (черный фон будет 0, остальные пиксели — 255)
    inverted_mask = cv.bitwise_not(mask)
    plt.figure(figsize=(6, 6))
    plt.imshow(inverted_mask, cmap='gray')
    plt.title("Маска по контурам")
    plt.axis('off')
    plt.show()

    # Применяем маску к каждому каналу
    masked_bands = [cv.bitwise_and(stacked_image[i], stacked_image[i], mask=mask) for i in range(bands)]

    # Находим контуры полезной области
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("Все данные на изображении пустые или черные.")

    # # Создаем точную маску на основе контуров
    # precise_mask = np.zeros_like(mask, dtype=np.uint8)
    # cv.drawContours(precise_mask, contours, -1, color=255, thickness=cv.FILLED)
    #
    # # Применяем точную маску к каждому каналу
    # final_masked_bands = [cv.bitwise_and(masked_band, masked_band, mask=precise_mask) for masked_band in masked_bands]
    #
    # # Создаем итоговый массив с прозрачными областями
    # transparent_image = np.stack(final_masked_bands, axis=0)  # (bands, height, width)

    # # Находим контуры полезной области и обрезаем
    # contour = np.vstack(contours)
    # x, y, w, h = cv.boundingRect(contour)
    #
    # cropped_image = transparent_image[:, y:y + h, x:x + w]
    #
    # # Обновляем геопривязку
    # new_transform = (
    #     transform[0] + x * transform[1],  # Новое начало по X
    #     transform[1],
    #     transform[2],
    #     transform[3] + y * transform[5],  # Новое начало по Y
    #     transform[4],
    #     transform[5],
    # )
    #
    # print(f"Форма результата: {cropped_image.shape}")
    #
    # return cropped_image, new_transform, projection
    # Создаем точную маску на основе контуров
    precise_mask = np.zeros_like(mask, dtype=np.uint8)
    cv.drawContours(precise_mask, contours, -1, color=255, thickness=cv.FILLED)

    # Применяем точную маску к каждому каналу
    final_masked_bands = [cv.bitwise_and(stacked_image[i], stacked_image[i], mask=precise_mask) for i in range(bands)]

    # Преобразуем итоговый результат с прозрачным фоном
    transparent_image = np.stack(final_masked_bands, axis=0)  # (bands, height, width)

    # Убираем черный фон, сохраняя только область внутри маски
    indices = np.where(precise_mask > 0)  # Координаты всех белых пикселей
    x_min, x_max = np.min(indices[1]), np.max(indices[1])  # Минимальные и максимальные значения по ширине
    y_min, y_max = np.min(indices[0]), np.max(indices[0])  # Минимальные и максимальные значения по высоте

    # plt.imshow(mask, cmap='gray')
    # plt.axvline(x=x_min, color='r')
    # plt.axvline(x=x_max, color='r')
    # plt.axhline(y=y_min, color='r')
    # plt.axhline(y=y_max, color='r')
    # plt.show()

    # Обрезаем изображение по этим координатам
    cropped_image = transparent_image[:, y_min:y_max + 1, x_min:x_max + 1]

    # plt.figure(figsize=(8, 8))
    # plt.imshow(cropped_image.transpose(1, 2, 0))  # Перекладываем оси для отображения
    # plt.title("Обрезанное изображение")
    # plt.axis('off')
    # plt.show()

    # Обновляем геопривязку
    new_transform = (
        transform[0] + x_min * transform[1],  # Новое начало по X
        transform[1],
        transform[2],
        transform[3] + y_min * transform[5],  # Новое начало по Y
        transform[4],
        transform[5],
    )

    print(f"Форма обрезанного изображения: {cropped_image.shape}")

    return cropped_image, new_transform, projection





# Функция для извлечения данных каналов
def get_channels(image, bands):
    """
    Извлекает данные всех каналов из изображения.
    Работает как с GDAL Dataset, так и с numpy.ndarray.

    :param image: GDAL Dataset или numpy.ndarray
    :param bands: Количество каналов
    :return: список массивов каналов (каждый канал — numpy.ndarray)
    """

    channels = []

    if isinstance(image, np.ndarray):
        # Если image — это уже массив (numpy.ndarray)
        if image.shape[0] != bands:
            raise ValueError("Количество каналов в массиве не совпадает с заданным bands.")
        channels = [image[i] for i in range(bands)]
    else:
        # Если image — это GDAL Dataset
        for i in range(1, bands + 1):
            band = image.GetRasterBand(i)
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


# def classify_evi(ndvi):
#     """
#     Классифицирует NDVI по категориям.
#
#     :param ndvi: Массив NDVI значений
#     :return: Массив категорий NDVI
#     """
#     # classification = np.zeros_like(ndvi, dtype=np.uint8)
#     classification_ndvi = np.zeros_like(ndvi)
#
#     # Условия классификации
#     classification_ndvi[(ndvi >= -1) & (ndvi < 0)] = 1  # Вода
#     classification_ndvi[(ndvi >= 0) & (ndvi < 0.2)] = 2  # Слабая растительность
#     classification_ndvi[(ndvi >= 0.2) & (ndvi < 0.6)] = 3  # Умеренная растительность
#     classification_ndvi[(ndvi >= 0.6)] = 4  # Плотная растительность
#
#     return classification_ndvi


# Функция для контрастного растяжения
def rescale_intensity(input_band):
    """
    Применяет контрастное растяжение к каналу с помощью гистограммы.

    :param input_band: Массив данных канала
    :return: Массив с растянутым контрастом
    """
    # return exposure.rescale_intensity(input_band, in_range='image', out_range=(0, 1))
    return exposure.rescale_intensity(input_band, in_range='image', out_range=(0, 1))


# Функция для контрастного растяжения
def stretch_contrast(input_band):
    """
    Применяет контрастное растяжение к каналу с помощью гистограммы.

    :param input_band: Массив данных канала
    :return: Массив с растянутым контрастом
    """
    return exposure.equalize_hist(input_band)




# Основной скрипт
if __name__ == "__main__":
    # file_path = 'data/multiband_imagery.tif'
    file_path = '050160619050_01_P001_MUL/22MAR06104502-M3DS_R3C5-050160619050_01_P001.TIF'
    try:
        metadata = extract_metadata(file_path)
        print("")
        print("Извлеченные метаданные:")
        print(f"Размеры: {metadata['rows']} строк, {metadata['cols']} столбцов")
        print("")
        print("Геопривязка:")
        for key, value in metadata['geo_transform'].items():
            print(f"  {key}: {value}")
        print("")
        print(f"Система координат: {metadata['projection']}")
    except FileNotFoundError as e:
        print(e)

    imagery_ds, num_bands = open_multiband_image(file_path)



    # cropped_image = remove_black_zones_and_save_simple(imagery_ds)
    # Применение функции
    image_with_alpha, new_transform, projection = remove_black_zones_and_save_simple(imagery_ds)

    # Извлечение каналов
    channels = get_channels(image_with_alpha, num_bands)


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

    # Демонстрация массива для канала Blue
    display_channel_array(Blue, "Blue")

    for channel_name in band_names:
        if channel_dict.get(channel_name) is None:
            print("")
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
              f"Min: {types_normalize.normalize_band_global_max(channel.min())}, "
              f"Max: {types_normalize.normalize_band_global_max(channel.max())}"
              )
    print("\n")

    # Информация о каналах линейная нормализация от 0 до 1
    print("Информация о каналах line_normalize:")
    for name, channel in channel_dict.items():
        print(
            f"{name} - "
            f"Min: {types_normalize.line_normalize(channel.min())}, "
            f"Max: {types_normalize.line_normalize(channel.max())}"
              )
    print("\n")

    # Информация о каналах normalize_with_delite_emissions
    print("Информация о каналах normalize_with_delite_emissions:")
    for name, channel in channel_dict.items():
        print(f"{name} - "
              f"Min: {types_normalize.normalize_with_delite_emissions(channel.min())}, "
              f"Max: {types_normalize.normalize_with_delite_emissions(channel.max())}"
              )
    print("\n")

    # # if "Red" in named_channels and "Green" in named_channels and "Blue" in named_channels:
    # if Red is not None and Green is not None and Blue is not None:
    #     visualizations_plot.plot_rgb(
    #         types_normalize.normalize_band_global_max(Red),
    #         types_normalize.normalize_band_global_max(Green),
    #         types_normalize.normalize_band_global_max(Blue),
    #     )
    #
    # if Red is not None and Green is not None and Blue is not None:
    #     visualizations_plot.plot_rgb(
    #         types_normalize.line_normalize(Red),
    #         types_normalize.line_normalize(Green),
    #         types_normalize.line_normalize(Blue),
    #     )
    #
    # if Red is not None and Green is not None and Blue is not None:
    #     visualizations_plot.plot_rgb(
    #         types_normalize.normalize_with_delite_emissions(Red),
    #         types_normalize.normalize_with_delite_emissions(Green),
    #         types_normalize.normalize_with_delite_emissions(Blue),
    #     )

    if Red is not None and Green is not None and Blue is not None:
        red1, green1, blue1 = \
            types_normalize.normalize_band_global_max(Red), \
            types_normalize.normalize_band_global_max(Green),\
            types_normalize.normalize_band_global_max(Blue)
        print("red1 shape:", red1.shape)
        print("green1 shape:", green1.shape)
        print("blue1 shape:", blue1.shape)
        # Простой вывод первого набора нормализованных данных
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(np.stack((red1, green1, blue1), axis=-1))
        plt.title("RGB (global max normalization)")

        red2, green2, blue2 = \
            types_normalize.line_normalize(Red), \
            types_normalize.line_normalize(Green), \
            types_normalize.line_normalize(Blue)
        red3, green3, blue3 = \
            types_normalize.normalize_with_delite_emissions(Red), \
            types_normalize.normalize_with_delite_emissions(Green), \
            types_normalize.normalize_with_delite_emissions(Blue)

        # Вызов функции для визуализации
        visualizations_plot.plot_quantity_three_rgb(
            red1, green1, blue1,
            red2, green2, blue2,
            red3, green3, blue3,
            titles=("Default Normalize (global_max)", "Line Normalize", "Clip Normalize")
        )
        # -----------------------------------Histograms---------------------------------------------------------------
        # Вызов функции для отображения гистограмм
        # visualizations_plot_histogram.plot_histograms_three(
        #     red1, green1, blue1,
        #     red2, green2, blue2,
        #     red3, green3, blue3,
        #     titles=("Default Normalize Histogram", "Line Normalize Histogram", "Clip Normalize Histogram")
        # )
        # -----------------------------------Histograms---------------------------------------------------------------

# -------------------------------------stretch_contrast---------------------------------------------------------------
    if Red is not None and Green is not None and Blue is not None:
        # Предположим, у вас есть три набора нормализованных RGB-каналов
        red1, green1, blue1 = \
            stretch_contrast(types_normalize.normalize_band_global_max(Red)),\
            stretch_contrast(types_normalize.normalize_band_global_max(Green)),\
            stretch_contrast(types_normalize.normalize_band_global_max(Blue))
        red2, green2, blue2 = \
            stretch_contrast(types_normalize.line_normalize(Red)), \
            stretch_contrast(types_normalize.line_normalize(Green)), \
            stretch_contrast(types_normalize.line_normalize(Blue))
        red3, green3, blue3 = \
            stretch_contrast(types_normalize.normalize_with_delite_emissions(Red)), \
            stretch_contrast(types_normalize.normalize_with_delite_emissions(Green)), \
            stretch_contrast(types_normalize.normalize_with_delite_emissions(Blue))

        # # Вызов функции для визуализации
        # visualizations_plot.plot_quantity_three_rgb(
        #     red1, green1, blue1,
        #     red2, green2, blue2,
        #     red3, green3, blue3,
        #     titles=("Default Normalize (global_max)", "Line Normalize", "Clip Normalize")
        # )
#         # -----------------------------------Histograms---------------------------------------------------------------
#         # Вызов функции для отображения гистограмм
#         # visualizations_plot_histogram.plot_histograms_three(
#         #     red1, green1, blue1,
#         #     red2, green2, blue2,
#         #     red3, green3, blue3,
#         #     titles=("Default Normalize Histogram", "Line Normalize Histogram", "Clip Normalize Histogram")
#         # )
#         visualizations_plot_histogram.plot_histograms_three_single_channel2(
#             red1, red2, red3,
#             titles=("red1", "red2", "red3"),
#             bins=128, color='green', alpha=0.7
#         )
#         # -----------------------------------Histograms---------------------------------------------------------------
# -------------------------------------stretch_contrast---------------------------------------------------------------

    # if NIR is not None:
    #     plot_band(normalize_band_global_max(NIR), "Ближний инфракрасный (NIR) normalize_band_global_max")
    #
    # if NIR is not None:
    #     plot_band(line_normalize(NIR), "Ближний инфракрасный (NIR) line_normalize")
    #
    # if NIR is not None:
    #     plot_band(normalize_with_delite_emissions(NIR), "Ближний инфракрасный (NIR) normalize_with_delite_emissions")

    if NIR is not None:
        # Используем NIR канал (и его модификации)
        nir1 = types_normalize.normalize_band_global_max(NIR)  # Исходный канал
        nir2 = types_normalize.line_normalize(NIR)  # Линейная нормализация
        nir3 = types_normalize.normalize_with_delite_emissions(NIR)  # Нормализация с удалением выбросов

        # # Вызов функции для визуализации
        # visualizations_plot.plot_quantity_three_single_channel(
        #     nir1, nir2, nir3,
        #     titles=("NIR Original", "NIR Line Normalize", "NIR Clip Normalize"),
        #     colormap='viridis'  # Например, можно использовать цветовую карту 'viridis'
        # )
        # -----------------------------------Histograms---------------------------------------------------------------
        # # Построение гистограмм
        # visualizations_plot_histogram.plot_histograms_three_single_channel(
        #     nir1, nir2, nir3,
        #     titles=("NIR Original", "NIR Line Normalize", "NIR Clip Normalize")
        # )
        # -----------------------------------Histograms---------------------------------------------------------------

# ---------------------------------------arvi_normalize---------------------------------------------------------------
    if Red is not None and Green is not None and Blue is not None and NIR is not None:
        # Вычисление NDVI
        arvi_normalize_band_global_max = arvi.calculate_arvi(
            types_normalize.normalize_band_global_max(NIR),
            types_normalize.normalize_band_global_max(Red),
            types_normalize.normalize_band_global_max(Blue)
        )
        arvi_line_normalize = arvi.calculate_arvi(
            types_normalize.line_normalize(NIR),
            types_normalize.line_normalize(Red),
            types_normalize.line_normalize(Blue)
        )
        arvi_normalize_with_delite_emissions = arvi.calculate_arvi(
            types_normalize.normalize_with_delite_emissions(NIR),
            types_normalize.normalize_with_delite_emissions(Red),
            types_normalize.normalize_with_delite_emissions(Blue)
        )

        # visualizations_plot.plot_arvi(arvi_normalize_band_global_max)
        # visualizations_plot.plot_arvi(arvi_line_normalize)
        # visualizations_plot.plot_arvi(arvi_normalize_with_delite_emissions)

        '''Классификация ARVI'''
        arvi_classification_normalize_band_global_max = arvi.classify_arvi(arvi_normalize_band_global_max)
        arvi_classification_line_normalize = arvi.classify_arvi(arvi_line_normalize)
        arvi_classification_normalize_with_delite_emissions = arvi.classify_arvi(arvi_normalize_with_delite_emissions)
        # visualizations_plot.plot_classification_ndvi_with_labels(arvi_classification_normalize_band_global_max)
        # visualizations_plot.plot_classification_ndvi_with_labels(arvi_classification_line_normalize)
        # visualizations_plot.plot_classification_ndvi_with_labels(arvi_classification_normalize_with_delite_emissions)

        '''Визуализация 3 результатов ARVI'''
        # visualizations_plot.plot_quantity_three_single_channel(
        #     arvi_normalize_band_global_max, arvi_line_normalize,
        #     arvi_normalize_with_delite_emissions,
        #     titles=("Arvi Original", "Arvi Line Normalize",
        #             "Arvi Clip Normalize"),
        #     colormap='RdYlGn'  # Например, можно использовать цветовую карту 'viridis'
        # )

        '''Гистограмма ARVI'''
        # # тест гистограммы (которая бесполезная)
        # visualizations_plot_histogram.plot_histogram(arvi_classification_normalize_band_global_max,
        #                                              title="Arvi Histogram", bins=30, color="red", range=(-1, 1))

# ---------------------------------------arvi_normalize---------------------------------------------------------------

# ----------------------------------------EVI_normalize---------------------------------------------------------------
    if Red is not None and Green is not None and Blue is not None and NIR is not None:
        # Вычисление EVI
        evi_normalize_band_global_max = evi.calculate_evi(
            types_normalize.normalize_band_global_max(NIR),
            types_normalize.normalize_band_global_max(Red),
            types_normalize.normalize_band_global_max(Blue)
        )
        evi_line_normalize = evi.calculate_evi(
            types_normalize.line_normalize(NIR),
            types_normalize.line_normalize(Red),
            types_normalize.line_normalize(Blue)
        )
        evi_normalize_with_delite_emissions = evi.calculate_evi(
            types_normalize.normalize_with_delite_emissions(NIR),
            types_normalize.normalize_with_delite_emissions(Red),
            types_normalize.normalize_with_delite_emissions(Blue))

        # visualizations_plot.plot_evi(evi_normalize_band_global_max)
        # visualizations_plot.plot_evi(evi_line_normalize)
        # visualizations_plot.plot_evi(evi_normalize_with_delite_emissions)

        '''Классификация EVI'''
        evi_classification_normalize_band_global_max = evi.classify_evi(evi_normalize_band_global_max)
        evi_classification_line_normalize = evi.classify_evi(evi_line_normalize)
        evi_classification_normalize_with_delite_emissions = evi.classify_evi(evi_normalize_with_delite_emissions)
        # visualizations_plot.plot_classification_evi_with_labels(evi_classification_normalize_band_global_max)
        # visualizations_plot.plot_classification_evi_with_labels(evi_classification_line_normalize)
        # visualizations_plot.plot_classification_evi_with_labels(evi_classification_normalize_with_delite_emissions)

        # visualizations_plot.plot_quantity_three_single_channel(
        #     evi_normalize_band_global_max, evi_line_normalize, evi_normalize_with_delite_emissions,
        #     titles=("EVI Original", "EVI Line Normalize", "EVI Clip Normalize"),
        #     colormap='viridis'  # Например, можно использовать цветовую карту 'viridis'
        # )

        '''Визуализация 3 результатов ARVI'''
        # visualizations_plot.plot_quantity_three_single_channel(
        #     evi_classification_normalize_band_global_max, evi_classification_line_normalize,
        #     evi_classification_normalize_with_delite_emissions,
        #     titles=("Evi Original", "Evi Line Normalize",
        #             "Evi Clip Normalize"),
        #     colormap='viridis'  # Например, можно использовать цветовую карту 'viridis'
        # )

        # -----------------------------------Histograms---------------------------------------------------------------
        # # Построение гистограмм
        # visualizations_plot_histogram.plot_histograms_three_single_channel(
        #     ndvi_normalize_band_global_max, ndvi_line_normalize, ndvi_normalize_with_delite_emissions,
        #     titles=("NIR Original", "NIR Line Normalize", "NIR Clip Normalize")
        # )
        # -----------------------------------Histograms---------------------------------------------------------------
# ---------------------------------------EVI_normalize---------------------------------------------------------------

# ---------------------------------------ndvi_normalize---------------------------------------------------------------
    if Red is not None and Green is not None and Blue is not None and NIR is not None:
        # Вычисление NDVI
        ndvi_normalize_band_global_max = ndvi.calculate_ndvi(types_normalize.normalize_band_global_max(NIR),
                                                        types_normalize.normalize_band_global_max(Red))
        ndvi_line_normalize = ndvi.calculate_ndvi(types_normalize.line_normalize(NIR), types_normalize.line_normalize(Red))
        ndvi_normalize_with_delite_emissions = ndvi.calculate_ndvi(types_normalize.normalize_with_delite_emissions(NIR),
                                                              types_normalize.normalize_with_delite_emissions(Red))

        # visualizations_plot.plot_ndvi(ndvi_normalize_band_global_max)
        # visualizations_plot.plot_ndvi(ndvi_line_normalize)
        # visualizations_plot.plot_ndvi(ndvi_normalize_with_delite_emissions)

        '''Визуализация 3 результатов Ndvi'''
        # visualizations_plot.plot_quantity_three_single_channel(
        #     ndvi_normalize_band_global_max, ndvi_line_normalize, ndvi_normalize_with_delite_emissions,
        #     titles=("Ndvi Original", "Ndvi Line Normalize", "Ndvi Clip Normalize"),
        #     colormap='RdYlGn'  # Например, можно использовать цветовую карту 'viridis'
        # )
        # -----------------------------------Histograms---------------------------------------------------------------
        # # Построение гистограмм
        # visualizations_plot_histogram.plot_histograms_three_single_channel(
        #     ndvi_normalize_band_global_max, ndvi_line_normalize, ndvi_normalize_with_delite_emissions,
        #     titles=("NIR Original", "NIR Line Normalize", "NIR Clip Normalize")
        # )
        # -----------------------------------Histograms---------------------------------------------------------------
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Классификация NDVI
        ndvi_classification_normalize_band_global_max = ndvi.classify_ndvi(ndvi_normalize_band_global_max)
        ndvi_classification_line_normalize = ndvi.classify_ndvi(ndvi_line_normalize)
        ndvi_classification_normalize_with_delite_emissions = ndvi.classify_ndvi(ndvi_normalize_with_delite_emissions)

        # visualizations_plot.plot_classification_ndvi_with_labels(ndvi_classification_normalize_band_global_max)
        # visualizations_plot.plot_classification_ndvi_with_labels(ndvi_classification_line_normalize)
        # visualizations_plot.plot_classification_ndvi_with_labels(ndvi_classification_normalize_with_delite_emissions)

        '''Визуализация 3 результатов Ndvi'''
        # visualizations_plot.plot_quantity_three_single_channel(
        #     ndvi_classification_normalize_band_global_max, ndvi_classification_line_normalize,
        #     ndvi_classification_normalize_with_delite_emissions,
        #     titles=("Ndvi_classification Original", "Ndvi_classification Line Normalize",
        #             "Ndvi_classification Clip Normalize"),
        #     colormap='viridis'  # Например, можно использовать цветовую карту 'viridis'
        # )
        # -----------------------------------Histograms---------------------------------------------------------------
        # # Построение гистограмм
        # visualizations_plot_histogram.plot_histograms_three_single_channel(
        #     ndvi_classification_normalize_band_global_max, ndvi_classification_line_normalize,
        #     ndvi_classification_normalize_with_delite_emissions,
        #     titles=("Ndvi Original", "Ndvi Line Normalize", "Ndvi Clip Normalize")
        # )
        # -----------------------------------Histograms---------------------------------------------------------------


        rgb = np.dstack([types_normalize.normalize_with_delite_emissions(Red),
                         types_normalize.normalize_with_delite_emissions(Green),
                         types_normalize.normalize_with_delite_emissions(Blue)])

        # visualizations_plot.plot_quantity_three_single_channel(
        #     rgb, normalize_with_delite_emissions(NIR),
        #     ndvi_classification_normalize_band_global_max,
        #     titles=("Rgb normalize_with_delite_emissions", "NIR line_normalize",
        #             "Ndvi_classification global_max"),
        #     colormap='viridis'  # Например, можно использовать цветовую карту 'viridis'
        # )

    # 3 rgb -> 2 nir -> 1 ndvi? (гистограма говорит 3) -> непонятно classification ->

# ______________________________________________________________________________________________________________________
#         # Обнаружение облаков
#         cloud_mask = cloud_processing.detect_clouds_spectral(ndvi_normalize_with_delite_emissions)
#
#         # Считать все каналы в 3D массив
#         dataset = gdal.Open(file_path)
#         image = np.stack([dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)], axis=2)
#         # # Удалить облака
#         # clean_image = cloud_processing.remove_clouds(image, cloud_mask, nodata_value=0)
#
#         # Убедимся, что cloud_mask — это логическая маска (1 - облака, 0 - чисто)
#         cloud_mask2 = cloud_mask.astype(bool)
#         # # Заполнить облачные области
#         # filled_image = cloud_processing.fill_clouds(image, cloud_mask)
#         filled_image, metadata = cloud_processing.fill_clouds_with_metadata(image, cloud_mask2, new_transform, projection)
#
#         asdas = stretch_contrast(filled_image)
#
#         # Маска облаков
#         visualizations_plot.plot_evi(filled_image)
#         rgb_image = cloud_processing.extract_rgb(asdas)
#         plt.imshow(rgb_image)
#         plt.axis('off')
#         plt.show()
#
#         # # inverted_mask = cv.bitwise_not(cloud_mask)
#         # plt.figure(figsize=(6, 6))
#         # plt.imshow(filled_image, cmap='gray')
#         # plt.title("Маска по контурам")
#         # plt.axis('off')
#         # plt.show()
#
#
#         # ПОПЫТКА НОМЕР 2
#         brightness = (Red + Green + Blue) / 3
#         ndvi = (NIR - Red) / (NIR + Red + 1e-6)  # NDVI расчет
#         # # Удаляем облака
#         # processed_data = cloud_processing.remove_clouds1(brightness, ndvi, np.array(channels))
#         # cloud_processing.display_image(processed_data)
#         # processed_data2 = cloud_processing.remove_clouds_combined(brightness, ndvi, np.array(channels))
#         # cloud_processing.display_image(processed_data2)
#
#
#         # cloud_mask3 = (brightness > 2000) & (ndvi < 0.1)
#         # # Вычисляем маску облаков
#         cloud_mask3 = cloud_processing.detect_clouds_combined(
#             ndvi=ndvi,
#             brightness=brightness,
#             swir=SWIR1,  # Используем SWIR1, если доступно
#             ndvi_threshold=0.1,  # Увеличиваем порог NDVI для более точного результата
#             brightness_threshold=2000,
#             swir_threshold=0.5  # Порог для SWIR
#         )
#         '''интересный результат'''
#         # spatial_data = cloud_processing.spatial_interpolation(np.array(channels), cloud_mask)
#         # cloud_processing.display_image(spatial_data)
#         """!!!ТУТ ОСТАНОВИЛСЯ В РАБОТЕ С УДАЛЕНИЕМ ОБЛАВКОВ!!!"""
#         spatial_data = cloud_processing.enhanced_spatial_interpolation(np.array(channels), cloud_mask3)
#         cloud_processing.display_image(spatial_data)
#
#         # geometric_data = cloud_processing.geometric_reconstruction(np.array(channels), cloud_mask)
#         # cloud_processing.display_image(geometric_data)
#
#         # texture_data = cloud_processing.texture_analysis_reconstruction(np.array(channels), cloud_mask)
#         # cloud_processing.display_image(texture_data)

# ----------------------------------------ndvi_normalize---------------------------------------------------------------

# ---------------------------------------ndwi_normalize---------------------------------------------------------------
    if NIR is not None and Green is not None:
        # Вычисление NDVI
        ndwi_normalize_band_global_max = ndwi.calculate_ndwi(
            types_normalize.normalize_band_global_max(NIR),
            types_normalize.normalize_band_global_max(Green)
        )
        ndwi_line_normalize = ndwi.calculate_ndwi(
            types_normalize.line_normalize(NIR),
            types_normalize.line_normalize(Green)
        )
        ndwi_normalize_with_delite_emissions = ndwi.calculate_ndwi(
            types_normalize.normalize_with_delite_emissions(NIR),
            types_normalize.normalize_with_delite_emissions(Green)
        )

        # visualizations_plot.plot_ndwi(ndwi_normalize_band_global_max)
        # visualizations_plot.plot_ndwi(ndwi_line_normalize)
        # visualizations_plot.plot_ndwi(ndwi_normalize_with_delite_emissions)

        '''Классификация NDWI'''
        ndwi_classification_normalize_band_global_max = ndwi.classify_ndwi(ndwi_normalize_band_global_max)
        ndwi_classification_line_normalize = ndwi.classify_ndwi(ndwi_line_normalize)
        ndwi_classification_normalize_with_delite_emissions = ndwi.classify_ndwi(ndwi_normalize_with_delite_emissions)
        # visualizations_plot.plot_classification_ndwi_with_labels(ndwi_classification_normalize_band_global_max)
        # visualizations_plot.plot_classification_ndwi_with_labels(ndwi_classification_line_normalize)
        # visualizations_plot.plot_classification_ndwi_with_labels(ndwi_classification_normalize_with_delite_emissions)
# ---------------------------------------ndwi_normalize---------------------------------------------------------------

# ---------------------------------------savi_normalize---------------------------------------------------------------
    if NIR is not None and Red is not None:
        # Вычисление NDVI
        savi_normalize_band_global_max = savi.calculate_savi(
            types_normalize.normalize_band_global_max(NIR),
            types_normalize.normalize_band_global_max(Red)
        )
        savi_line_normalize = savi.calculate_savi(
            types_normalize.line_normalize(NIR),
            types_normalize.line_normalize(Red)
        )
        savi_normalize_with_delite_emissions = savi.calculate_savi(
            types_normalize.normalize_with_delite_emissions(NIR),
            types_normalize.normalize_with_delite_emissions(Red)
        )

        # visualizations_plot.plot_savi(savi_normalize_band_global_max)
        # visualizations_plot.plot_savi(savi_line_normalize)
        # visualizations_plot.plot_savi(savi_normalize_with_delite_emissions)

        '''Классификация SAVI'''
        savi_classification_normalize_band_global_max = savi.classify_savi(savi_normalize_band_global_max)
        savi_classification_line_normalize = savi.classify_savi(savi_line_normalize)
        savi_classification_normalize_with_delite_emissions = savi.classify_savi(savi_normalize_with_delite_emissions)
        # visualizations_plot.plot_classification_savi_with_labels(savi_classification_normalize_band_global_max)
        # visualizations_plot.plot_classification_savi_with_labels(savi_classification_line_normalize)
        # visualizations_plot.plot_classification_savi_with_labels(savi_classification_normalize_with_delite_emissions)
# ---------------------------------------savi_normalize---------------------------------------------------------------


    if Red is not None and Green is not None and Blue is not None:
        # output_png = "output/test_tif_in_png.png"
        # output_png = "output/test_rgb_to_rgba.png"
        output_png = "output/rgba_black_transparent.png"
        # output_png = "output/rgba_region_transparent.png"


        def is_black(pixel_array):
            """Условие: Сделать прозрачным пиксели, которые близки к чёрному."""
            threshold = 0.1  # Порог, ниже которого пиксель считается чёрным
            return np.all(pixel_array < threshold, axis=-1)


        def is_in_region(pixel_array):
            """Условие: Сделать прозрачной центральную область изображения."""
            rows, cols = pixel_array.shape[:2]
            row_start, row_end = rows // 4, 3 * rows // 4
            col_start, col_end = cols // 4, 3 * cols // 4

            mask = np.zeros((rows, cols), dtype=bool)
            mask[row_start:row_end, col_start:col_end] = True
            return mask


        alpha_value = 0.1  # Канал прозрачности
        rgb = np.dstack([types_normalize.normalize_with_delite_emissions(Red),
                         types_normalize.normalize_with_delite_emissions(Green),
                         types_normalize.normalize_with_delite_emissions(Blue)])
        alpha_gradient = np.linspace(0, 1, rgb.shape[0])[:, None] * np.ones((1, rgb.shape[1]))

        # # Конвертация массива в PNG
        # try:
        #     # result_file = tif_in_png.convert_normalized_array_to_png(rgb, output_png)
        #     # result_file = tif_in_png.convert_rgb_to_rgba_and_save(rgb, output_png, alpha_gradient)
        #     result_file = tif_in_png.convert_rgb_to_rgba_with_condition(rgb, output_png, is_black)
        #     # result_file = tif_in_png.convert_rgb_to_rgba_with_condition(rgb, output_png, is_in_region)
        #     print(f"Изображение сохранено в: {result_file}")
        # except ValueError as e:
        #     print(f"Ошибка: {e}")






    """добавить работу с другими каналами"""

    for swir_name in [SWIR1, SWIR2, SWIR3]:                                                       #????????????????????
        if swir_name is not None:
            visualizations_plot.plot_band(types_normalize.normalize_band_global_max(swir_name),
                                          f"{swir_name} (Средний инфракрасный)")

# ---------------------------------------mndwi_normalize---------------------------------------------------------------
        if NIR is not None and SWIR1 or SWIR2 or SWIR3 is not None:
            # Вычисление NDVI
            mndwi_normalize_band_global_max = mndwi.calculate_mndwi(types_normalize.normalize_band_global_max(NIR),
                                                                    types_normalize.normalize_band_global_max(SWIR1)
                                                                    )
            mndwi_line_normalize = mndwi.calculate_mndwi(types_normalize.line_normalize(NIR),
                                                         types_normalize.line_normalize(SWIR1)
                                                         )
            mndwi_normalize_with_delite_emissions = mndwi.calculate_mndwi(
                types_normalize.normalize_with_delite_emissions(NIR),
                types_normalize.normalize_with_delite_emissions(SWIR1)
            )

            # visualizations_plot.plot_ndvi(mndwi_normalize_band_global_max)
            # visualizations_plot.plot_ndvi(mndwi_line_normalize)
            # visualizations_plot.plot_ndvi(mndwi_normalize_with_delite_emissions)
# ---------------------------------------mndwi_normalize---------------------------------------------------------------

# ---------------------------------------mndwi_normalize---------------------------------------------------------------
        if Green is not None and Red is not None and NIR is not None and SWIR1 or SWIR2 or SWIR3 is not None:
            # Вычисление NDVI
            mndwi_normalize_band_global_max = mndwi.calculate_mndwi(types_normalize.normalize_band_global_max(NIR),
                                                                    types_normalize.normalize_band_global_max(SWIR1)
                                                                    )
            mndwi_line_normalize = mndwi.calculate_mndwi(types_normalize.line_normalize(NIR),
                                                         types_normalize.line_normalize(SWIR1)
                                                         )
            mndwi_normalize_with_delite_emissions = mndwi.calculate_mndwi(
                types_normalize.normalize_with_delite_emissions(NIR),
                types_normalize.normalize_with_delite_emissions(SWIR1)
            )

            # visualizations_plot.plot_ndvi(mndwi_normalize_band_global_max)
            # visualizations_plot.plot_ndvi(mndwi_line_normalize)
            # visualizations_plot.plot_ndvi(mndwi_normalize_with_delite_emissions)
# ---------------------------------------mndwi_normalize---------------------------------------------------------------

    if PAN is not None:
        visualizations_plot.plot_band(types_normalize.normalize_band_global_max(PAN),
                                      "Панхроматический канал (PAN)")

    if TIR is not None:
        visualizations_plot.plot_band(types_normalize.normalize_band_global_max(TIR),
                                      "Тепловое инфракрасное излучение (TIR)")
