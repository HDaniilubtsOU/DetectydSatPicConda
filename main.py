from osgeo import gdal
from osgeo import ogr, osr, gdal_array, gdalconst
from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from skimage import exposure
import textwrap


# нужно написать функции для 3 видов спутников:
# 1) Спутники наблюдения за Землей (Landsat: 9 каналов) 2) Метеорологические спутники (NOAA AVHRR: 5 кан.)
# 3) Гиперспектральные спутники (Hyperion: до 220 спектральных каналов в узких диапазонах,
# охватывающих видимый и инфракрасный спектр)


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

    # Вычисляем NDVI с защитой от деления на ноль
    ndvi = (nir - red) / (nir + red + epsilon)

    # # Обработка деления на ноль (замена NaN на 0)
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


# Усиленный вегетационный индекс (EVI) используется для улучшения оценки плотности растительности,
# устраняя атмосферные эффекты и влияние почвы.
# def calculate_evi(nir, red, blue, G=2.5, C1=6.0, C2=7.5, L=1.0):
def calculate_evi(nir, red, blue, G=2.5, C1=6.0, C2=7.5, L=1.0):
    """
    Рассчитывает индекс EVI.

    :param nir: Ближний инфракрасный канал.
    :param red: Красный канал.
    :param blue: Синий канал.
    :param G: Коэффициент усиления.
    :param C1: Коэффициент для коррекции атмосферы (Red).
    :param C2: Коэффициент для коррекции атмосферы (Blue).
    :param L: Поправочный коэффициент.
    :return: EVI массив.
    """
    nir = nir.astype(float)
    red = red.astype(float)
    blue = blue.astype(float)
    denominator = (nir + C1 * red - C2 * blue + L)
    evi = G * (nir - red) / denominator

    # Обработка NaN и бесконечных значений
    evi[np.isnan(evi)] = 0
    evi[np.isinf(evi)] = 0
    return evi


def classify_evi(ndvi):
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
    classification_ndvi[(ndvi >= 0.2) & (ndvi < 0.6)] = 3  # Умеренная растительность
    classification_ndvi[(ndvi >= 0.6)] = 4  # Плотная растительность

    return classification_ndvi


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


def plot_ndvi(ndvi, title="Ndvi Map"):
    """
    Визуализирует NDVI.

    :param ndvi: Массив NDVI.
    :param title: Заголовок графика.
    """
    # plt.imshow(ndvi, cmap='jet', vmin=-1, vmax=1)          # крутой стиль, не нрав
    # plt.imshow(ndvi, cmap='BrBG', vmin=-1, vmax=1)         # тоже неплох
    plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(label="Ndvi Value")
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_classification_ndvi(classified_ndvi, title="NDVI Classification"):
    """
    Визуализирует классифицированный NDVI.

    :param classified_ndvi: Классифицированный массив NDVI.
    :param title: Заголовок графика.
    """
    cmap = plt.get_cmap("Set3", 4)
    # cmap = colormaps.get_cmap("Set3").resampled(4)
    plt.imshow(classified_ndvi, cmap=cmap, vmin=1, vmax=4)
    plt.colorbar(ticks=[1, 2, 3, 4], label="Classes")
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_evi(evi, title="Evi Map"):
    """
    Визуализирует Evi.

    :param evi: Массив NDVI.
    :param title: Заголовок графика.
    """
    plt.imshow(evi, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(label="Evi Value")
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_classification_evi(classified_evi, title="NDVI Classification"):
    """
    Визуализирует классифицированный NDVI.

    :param classified_evi: Классифицированный массив NDVI.
    :param title: Заголовок графика.
    """
    cmap = plt.get_cmap("Set3", 4)
    # cmap = colormaps.get_cmap("Set3").resampled(4)
    plt.imshow(classified_evi, cmap=cmap, vmin=1, vmax=4)
    plt.colorbar(ticks=[1, 2, 3, 4], label="Classes")
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_quantity_three_single_channel(channel1, channel2, channel3,
                                       titles=("Channel 1", "Channel 2", "Channel 3"),
                                       colormap='gray'):
    """
    Визуализирует три отдельных канала в одном окне.

    :param channel1: Первый канал (например, NIR)
    :param channel2: Второй канал (например, NIR обработанный)
    :param channel3: Третий канал (например, NIR с фильтром)
    :param titles: Кортеж заголовков для каждого изображения
    :param colormap: Цветовая карта для отображения каналов (по умолчанию 'gray')
    """

    # Создание фигуры и трёх подграфиков
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 строка, 3 столбца

    # Первое изображение
    axs[0].imshow(channel1, cmap=colormap)
    axs[0].set_title(titles[0])
    axs[0].axis('off')

    # Второе изображение
    axs[1].imshow(channel2, cmap=colormap)
    axs[1].set_title(titles[1])
    axs[1].axis('off')

    # Третье изображение
    axs[2].imshow(channel3, cmap=colormap)
    axs[2].set_title(titles[2])
    axs[2].axis('off')

    # Отображение графиков
    plt.tight_layout()
    plt.show()


def plot_histograms_three_single_channel(pic1, pic2, pic3, titles=("Channel 1", "Channel 2", "Channel 3")):
    """
    Визуализирует гистограммы трёх одиночных каналов в одном окне.

    :param pic1: Первый изображение (например, NIR)
    :param pic2: Второй изображение (например, NIR обработанный)
    :param pic3: Третий изображение (например, NIR с фильтром)
    :param titles: Кортеж заголовков для каждого канала
    """
    fig_h, axs_h = plt.subplots(3, 1, figsize=(10, 15))  # 3 строки, 1 столбец

    # Вспомогательная функция для построения гистограммы одного изображения
    def plot_histogram(ax_h, pic, title):
        ax_h.hist(pic.ravel(), bins=256, color='yellow', alpha=0.5, label='yellow')
        ax_h.set_title(title)
        ax_h.set_xlabel('Intensity')
        ax_h.set_ylabel('Frequency')
        ax_h.legend()

    # Гистограмма для первого канала
    plot_histogram(axs_h[0], pic1, titles[0])

    # Гистограмма для второго канала
    plot_histogram(axs_h[1], pic2, titles[1])

    # Гистограмма для третьего канала
    plot_histogram(axs_h[2], pic3, titles[2])

    # Автоматическое выравнивание
    plt.tight_layout()
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
    # file_path = 'data/multiband_imagery.tif'
    file_path = '050160619050_01_P001_MUL/22MAR06104502-M3DS_R1C1-050160619050_01_P001.TIF'
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

    # # if "Red" in named_channels and "Green" in named_channels and "Blue" in named_channels:
    # if Red is not None and Green is not None and Blue is not None:
    #     plot_rgb(
    #         normalize_band_global_max(Red),
    #         normalize_band_global_max(Green),
    #         normalize_band_global_max(Blue),
    #     )
    #
    # if Red is not None and Green is not None and Blue is not None:
    #     plot_rgb(
    #         line_normalize(Red),
    #         line_normalize(Green),
    #         line_normalize(Blue),
    #     )
    #
    # if Red is not None and Green is not None and Blue is not None:
    #     plot_rgb(
    #         normalize_with_delite_emissions(Red),
    #         normalize_with_delite_emissions(Green),
    #         normalize_with_delite_emissions(Blue),
    #     )

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
        # -----------------------------------Histograms---------------------------------------------------------------
        # Вызов функции для отображения гистограмм
        # plot_histograms_three(
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

        # # Вызов функции для визуализации
        # plot_quantity_three_rgb(
        #     red1, green1, blue1,
        #     red2, green2, blue2,
        #     red3, green3, blue3,
        #     titles=("Default Normalize (global_max)", "Line Normalize", "Clip Normalize")
        # )
#         # -----------------------------------Histograms---------------------------------------------------------------
#         # Вызов функции для отображения гистограмм
#         # plot_histograms_three(
#         #     red1, green1, blue1,
#         #     red2, green2, blue2,
#         #     red3, green3, blue3,
#         #     titles=("Default Normalize Histogram", "Line Normalize Histogram", "Clip Normalize Histogram")
#         # )
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
        nir1 = normalize_band_global_max(NIR)  # Исходный канал
        nir2 = line_normalize(NIR)  # Линейная нормализация
        nir3 = normalize_with_delite_emissions(NIR)  # Нормализация с удалением выбросов

        # # Вызов функции для визуализации
        # plot_quantity_three_single_channel(
        #     nir1, nir2, nir3,
        #     titles=("NIR Original", "NIR Line Normalize", "NIR Clip Normalize"),
        #     colormap='viridis'  # Например, можно использовать цветовую карту 'viridis'
        # )
        # -----------------------------------Histograms---------------------------------------------------------------
        # Построение гистограмм
        # plot_histograms_three_single_channel(
        #     nir1, nir2, nir3,
        #     titles=("NIR Original", "NIR Line Normalize", "NIR Clip Normalize")
        # )
        # -----------------------------------Histograms---------------------------------------------------------------

# ---------------------------------------ndvi_normalize---------------------------------------------------------------
    if Red is not None and Green is not None and Blue is not None and NIR is not None:
        # Вычисление NDVI
        ndvi_normalize_band_global_max = calculate_ndvi(normalize_band_global_max(NIR), normalize_band_global_max(Red))
        ndvi_line_normalize = calculate_ndvi(line_normalize(NIR), line_normalize(Red))
        ndvi_normalize_with_delite_emissions = calculate_ndvi(normalize_with_delite_emissions(NIR),
                                                              normalize_with_delite_emissions(Red))

        # plot_ndvi(ndvi_normalize_band_global_max)
        # plot_ndvi(ndvi_line_normalize)
        # plot_ndvi(ndvi_normalize_with_delite_emissions)

        # plot_quantity_three_single_channel(
        #     ndvi_normalize_band_global_max, ndvi_line_normalize, ndvi_normalize_with_delite_emissions,
        #     titles=("Ndvi Original", "Ndvi Line Normalize", "Ndvi Clip Normalize"),
        #     colormap='viridis'  # Например, можно использовать цветовую карту 'viridis'
        # )
        # -----------------------------------Histograms---------------------------------------------------------------
        # # Построение гистограмм
        # plot_histograms_three_single_channel(
        #     ndvi_normalize_band_global_max, ndvi_line_normalize, ndvi_normalize_with_delite_emissions,
        #     titles=("NIR Original", "NIR Line Normalize", "NIR Clip Normalize")
        # )
        # -----------------------------------Histograms---------------------------------------------------------------

        # Классификация NDVI
        ndvi_classification_normalize_band_global_max = classify_ndvi(ndvi_normalize_band_global_max)
        ndvi_classification_line_normalize = classify_ndvi(ndvi_line_normalize)
        ndvi_classification_normalize_with_delite_emissions = classify_ndvi(ndvi_normalize_with_delite_emissions)

        plot_classification_ndvi(ndvi_classification_normalize_band_global_max)
        plot_classification_ndvi(ndvi_classification_line_normalize)
        plot_classification_ndvi(ndvi_classification_normalize_with_delite_emissions)

        # plot_quantity_three_single_channel(
        #     ndvi_classification_normalize_band_global_max, ndvi_classification_line_normalize,
        #     ndvi_classification_normalize_with_delite_emissions,
        #     titles=("Ndvi_classification Original", "Ndvi_classification Line Normalize",
        #             "Ndvi_classification Clip Normalize"),
        #     colormap='viridis'  # Например, можно использовать цветовую карту 'viridis'
        # )
        # -----------------------------------Histograms---------------------------------------------------------------
        # # Построение гистограмм
        # plot_histograms_three_single_channel(
        #     ndvi_classification_normalize_band_global_max, ndvi_classification_line_normalize,
        #     ndvi_classification_normalize_with_delite_emissions,
        #     titles=("Ndvi Original", "Ndvi Line Normalize", "Ndvi Clip Normalize")
        # )
        # -----------------------------------Histograms---------------------------------------------------------------

        rgb = np.dstack([normalize_with_delite_emissions(Red), normalize_with_delite_emissions(Green),
                         normalize_with_delite_emissions(Blue)])

        # plot_quantity_three_single_channel(
        #     rgb, normalize_with_delite_emissions(NIR),
        #     ndvi_classification_normalize_band_global_max,
        #     titles=("Rgb normalize_with_delite_emissions", "NIR line_normalize",
        #             "Ndvi_classification global_max"),
        #     colormap='viridis'  # Например, можно использовать цветовую карту 'viridis'
        # )

    # 3 rgb -> 2 nir -> 1 ndvi? (гистограма говорит 3) -> непонятно classification ->

# ----------------------------------------EVI_normalize---------------------------------------------------------------

# ----------------------------------------EVI_normalize---------------------------------------------------------------
    if Red is not None and Green is not None and Blue is not None and NIR is not None:
        # Вычисление EVI
        evi_normalize_band_global_max = calculate_evi(normalize_band_global_max(NIR), normalize_band_global_max(Red),
                                                      normalize_band_global_max(Blue))
        evi_line_normalize = calculate_evi(line_normalize(NIR), line_normalize(Red), line_normalize(Blue))
        evi_normalize_with_delite_emissions = calculate_evi(normalize_with_delite_emissions(NIR),
                                                            normalize_with_delite_emissions(Red),
                                                            normalize_with_delite_emissions(Blue))

        # plot_evi(evi_normalize_band_global_max)
        # plot_evi(evi_line_normalize)
        # plot_evi(evi_normalize_with_delite_emissions)

        evi_classification_normalize_band_global_max = classify_evi(evi_normalize_band_global_max)
        evi_classification_line_normalize = classify_evi(evi_line_normalize)
        evi_classification_normalize_with_delite_emissions = classify_evi(evi_normalize_with_delite_emissions)

        plot_classification_evi(evi_classification_normalize_band_global_max)
        plot_classification_evi(evi_classification_line_normalize)
        plot_classification_evi(evi_classification_normalize_with_delite_emissions)

        # plot_quantity_three_single_channel(
        #     evi_normalize_band_global_max, evi_line_normalize, evi_normalize_with_delite_emissions,
        #     titles=("EVI Original", "EVI Line Normalize", "EVI Clip Normalize"),
        #     colormap='viridis'  # Например, можно использовать цветовую карту 'viridis'
        # )

        # -----------------------------------Histograms---------------------------------------------------------------
        # # Построение гистограмм
        # plot_histograms_three_single_channel(
        #     ndvi_normalize_band_global_max, ndvi_line_normalize, ndvi_normalize_with_delite_emissions,
        #     titles=("NIR Original", "NIR Line Normalize", "NIR Clip Normalize")
        # )
        # -----------------------------------Histograms---------------------------------------------------------------
# ---------------------------------------ndvi_normalize---------------------------------------------------------------






    """добавить работу с другими каналами"""

    for swir_name in [SWIR1, SWIR2, SWIR3]:                                                       #????????????????????
        if swir_name is not None:
            plot_band(normalize_band_global_max(swir_name), f"{swir_name} (Средний инфракрасный)")

    if PAN is not None:
        plot_band(normalize_band_global_max(PAN), "Панхроматический канал (PAN)")

    if TIR is not None:
        plot_band(normalize_band_global_max(TIR), "Тепловое инфракрасное излучение (TIR)")

