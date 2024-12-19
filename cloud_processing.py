import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.transform import resize
import matplotlib.pyplot as plt


"""обнаружение облаков"""
# Каналы с информацией о облаках -> Некоторые спутники (например, Sentinel-2, Landsat 8)
# содержат специальные каналы или метаданные, которые предоставляют маску облачности,
# например, QA60 в Sentinel-2 или BQA в Landsat 8.

# метод удаления тонких облаков вслепую на основе одного изображения,
# получивший название cloud perception integrated fast Fourier convolutional network (CP-FFCN)


def remove_clouds1(brightness, ndvi, bands_data, brightness_threshold=2000, nir_threshold=0.3, swir=None,
                   swir_threshold=None):
    """
        Удаляет облака из изображения .tif на основе яркости и анализа спектра.

        Параметры:
        - brightness: ndarray - массив яркости.
        - ndvi: ndarray - массив значений NDVI.
        - bands_data: ndarray - 3D массив спектральных данных (каналы, высота, ширина).
        - brightness_threshold: int - порог яркости для выявления облаков.
        - nir_threshold: float - минимальное значение NDVI.
        - swir: ndarray - дополнительный канал SWIR для улучшенной фильтрации облаков.
        - swir_threshold: float - порог для SWIR канала, если он используется.

        Возвращает:
        - обработанный 3D массив (каналы, высота, ширина) с удалёнными облаками.
    """

    # Вычисляем основную маску облаков
    cloud_mask = (brightness > brightness_threshold) & (ndvi < nir_threshold)
    # Добавляем SWIR фильтрацию, если указано
    if swir is not None and swir_threshold is not None:
        cloud_mask |= (swir > swir_threshold)

    # Копируем данные для сохранения оригинала
    processed_bands = bands_data.copy()

    # # Применяем медианную замену для каждого канала
    # for i in range(processed_bands.shape[0]):
    #     band = processed_bands[i]
    #     band[cloud_mask] = np.median(band[~cloud_mask])
    #     processed_bands[i] = band

    # Применяем замену для каждого канала
    for i in range(processed_bands.shape[0]):
        # Преобразуем канал в float для работы с NaN
        band = processed_bands[i].astype(float)
        band[cloud_mask] = np.nan  # Заменяем облака на NaN

        # Используем фильтрацию для заполнения NaN
        from scipy.ndimage import generic_filter
        nan_mask = np.isnan(band)
        band[nan_mask] = generic_filter(band, np.nanmean, size=5, mode='mirror')[nan_mask]

        # Возвращаем преобразованный канал в исходный массив
        processed_bands[i] = band

    return processed_bands

def remove_clouds_combined(brightness, ndvi, bands_data, brightness_threshold=2000, nir_threshold=0.3, swir=None,
                           swir_threshold=None):
    """
    Удаляет облака, комбинируя спектральный анализ и пространственную интерполяцию.
    """
    # Спектральный анализ для обнаружения облаков
    cloud_mask = (brightness > brightness_threshold) & (ndvi < nir_threshold)
    if swir is not None and swir_threshold is not None:
        cloud_mask |= (swir > swir_threshold)

    # Копируем данные
    processed_bands = bands_data.copy()

    # Пространственная интерполяция
    for i in range(processed_bands.shape[0]):
        band = processed_bands[i].astype(float)
        band[cloud_mask] = np.nan  # Заменяем облака на NaN

        # Заполнение NaN средним значением соседей
        from scipy.ndimage import generic_filter
        nan_mask = np.isnan(band)
        band[nan_mask] = generic_filter(band, np.nanmean, size=5, mode='mirror')[nan_mask]

        processed_bands[i] = band

    return processed_bands

# Визуализация изображения из обработанных данных
def display_image(processed_bands):
    """
    Отображает изображение на основе обработанного массива спектральных данных.

    Параметры:
    - processed_bands: ndarray - 3D массив (каналы, высота, ширина).
    """
    # Проверяем, есть ли достаточное количество каналов для RGB
    if processed_bands.shape[0] >= 3:
        # Нормализуем каналы для корректного отображения
        def normalize_channel(channel):
            channel_min = channel.min()
            channel_max = channel.max()
            return (channel - channel_min) / (channel_max - channel_min + 1e-6)

        # Извлекаем и нормализуем каналы
        red = normalize_channel(processed_bands[2])  # Red
        green = normalize_channel(processed_bands[1])  # Green
        blue = normalize_channel(processed_bands[0])  # Blue

        # Собираем RGB изображение
        rgb_image = np.dstack((red, green, blue))

        # Отображаем изображение
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_image)
        plt.title("Изображение после удаления облаков (RGB)")
        plt.axis("off")
        plt.show()
    else:
        # Если меньше 3 каналов, отображаем первый канал в градациях серого
        plt.figure(figsize=(10, 10))
        plt.imshow(processed_bands[0], cmap='gray')
        plt.title("Изображение после удаления облаков (Градации серого)")
        plt.axis("off")
        plt.show()


def spatial_interpolation(bands_data, cloud_mask):
    """
    Пространственная интерполяция для восстановления данных облаков.
    """
    processed_bands = bands_data.copy()
    for i in range(processed_bands.shape[0]):
        band = processed_bands[i].astype(float)
        band[cloud_mask] = np.nan

        # Заменяем NaN на 0 (или медианное значение)
        band = np.nan_to_num(band, nan=np.nanmedian(band))
        # Преобразуем в uint8 для анализа текстур
        band_uint8 = band.astype(np.uint8)

        # Используем билинейную интерполяцию
        from scipy.interpolate import griddata
        y, x = np.indices(band_uint8.shape)
        valid_mask = ~np.isnan(band_uint8)
        band_uint8[cloud_mask] = griddata(
            (x[valid_mask], y[valid_mask]),
            band_uint8[valid_mask],
            (x[cloud_mask], y[cloud_mask]),
            method='linear'
        )
        processed_bands[i] = band_uint8

    return processed_bands


def enhanced_spatial_interpolation(bands_data, cloud_mask):
    """
    Улучшенная пространственная интерполяция для удаления облаков.
    """
    processed_bands = bands_data.copy()

    for i in range(processed_bands.shape[0]):
        band = processed_bands[i].astype(float)
        y, x = np.indices(band.shape)

        from scipy.interpolate import griddata
        from scipy.ndimage import generic_filter
        # Первый этап: грубая интерполяция (метод ближайших соседей)
        valid_mask = ~cloud_mask
        band[cloud_mask] = griddata(
            (x[valid_mask], y[valid_mask]),
            band[valid_mask],
            (x[cloud_mask], y[cloud_mask]),
            method='nearest'
        )

        # Второй этап: сглаживание медианным фильтром
        band = generic_filter(band, np.nanmedian, size=5, mode='mirror')

        # Третий этап: локальная интерполяция через билинейный метод
        band[cloud_mask] = griddata(
            (x[valid_mask], y[valid_mask]),
            band[valid_mask],
            (x[cloud_mask], y[cloud_mask]),
            method='linear'
        )

        # Заменяем NaN на медианное значение или 0 перед преобразованием
        band = np.nan_to_num(band, nan=np.nanmedian(band))

        # Приведение к исходному типу данных
        processed_bands[i] = band.astype(bands_data.dtype)

    return processed_bands


def geometric_reconstruction(bands_data, cloud_mask):
    """
    Геометрическая реконструкция данных.
    """
    processed_bands = bands_data.copy()
    for i in range(processed_bands.shape[0]):
        band = processed_bands[i].astype(float)
        band[cloud_mask] = np.nan

        # Используем фильтрацию для сглаживания данных
        from scipy.ndimage import median_filter
        band[cloud_mask] = median_filter(band, size=5)[cloud_mask]

        processed_bands[i] = band

    return processed_bands


def texture_analysis_reconstruction(bands_data, cloud_mask):
    """
    Восстановление данных на основе текстурного анализа.
    """
    from skimage.feature import graycomatrix, graycoprops
    processed_bands = bands_data.copy()

    for i in range(processed_bands.shape[0]):
        band = processed_bands[i].astype(float)
        band[cloud_mask] = np.nan

        # Заменяем NaN на 0 (или медианное значение, если требуется)
        band = np.nan_to_num(band, nan=0)

        # Преобразуем в uint8 для анализа текстур
        band_uint8 = band.astype(np.uint8)

        # Вычисляем текстурные характеристики
        glcm = graycomatrix(band_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]

        # Заполняем облачные пиксели значением текстурного контраста (или любым другим подходящим значением)
        band[cloud_mask] = contrast

        processed_bands[i] = band

    return processed_bands




# Методы анализа спектра
def detect_clouds_spectral(ndvi, ndvi_threshold=0):
    """
    Обнаружение облаков на основе спектрального анализа (NDVI).

    :param ndvi: Индекс NDVI (Normalized Difference Vegetation Index)
    :param ndvi_threshold: float, порог для классификации облаков (по умолчанию 0)
    :return: ndarray, маска облаков (1 - облака, 0 - без облаков)
    """

    # Создать маску облаков
    cloud_mask = ndvi < ndvi_threshold  # NDVI ниже порога указывает на облака
    return cloud_mask


def detect_clouds_combined(ndvi, brightness, swir=None, ndvi_threshold=0, brightness_threshold=2000, swir_threshold=None):
    """
    Обнаружение облаков на основе спектрального анализа, яркости и SWIR (если доступно).

    :param ndvi: ndarray, индекс NDVI.
    :param brightness: ndarray, уровень яркости.
    :param swir: ndarray, канал SWIR (может быть None).
    :param ndvi_threshold: float, порог NDVI для классификации облаков.
    :param brightness_threshold: int, порог яркости для классификации облаков.
    :param swir_threshold: float, порог SWIR для классификации облаков (если используется).
    :return: ndarray, маска облаков (True - облака, False - без облаков).
    """
    # Базовая маска на основе NDVI
    cloud_mask = ndvi < ndvi_threshold

    # Добавляем условие яркости
    cloud_mask |= (brightness > brightness_threshold)

    # Добавляем SWIR анализ, если он доступен
    if swir is not None and swir_threshold is not None:
        cloud_mask |= (swir > swir_threshold)

    return cloud_mask

# Алгоритмы машинного обучения


"""удаление облаков"""
# Замена пикселей облаков значением NODATA
def remove_clouds(image, cloud_mask, nodata_value=0):
    """
    Удаление облаков из изображения на основе маски облаков.

    :param image: ndarray, 3D массив спутникового изображения (каналы в последнем измерении)
    :param cloud_mask: ndarray, 2D маска облаков (1 - облака, 0 - без облаков)
    :param nodata_value: int/float, значение, которым заменить пиксели с облаками
    :return: ndarray, изображение с удаленными облаками
    """
    # Создаем копию изображения для модификации
    clean_image = np.copy(image)

    # Проверяем размерность
    if len(image.shape) == 2:  # Одноканальное изображение
        clean_image[cloud_mask] = nodata_value
    elif len(image.shape) == 3:  # Многоканальное изображение
        for i in range(image.shape[2]):  # Пройти по каждому каналу
            clean_image[:, :, i][cloud_mask] = nodata_value
    else:
        raise ValueError("Неподдерживаемая размерность изображения")

    return clean_image


# Интерполяция
def fill_clouds_with_metadata(image, cloud_mask, geotransform=None, projection=None):
    """
    Заполнение облаков с сохранением метаинформации изображения.

    :param image: ndarray, 3D массив спутникового изображения (каналы в последнем измерении).
    :param cloud_mask: ndarray, 2D маска облаков (1 - облака, 0 - без облаков).
    :param geotransform: tuple, геопривязка изображения (опционально).
    :param projection: str, проекция изображения в формате WKT (опционально).
    :return: tuple(ndarray, dict), (обработанное изображение, метаданные).
    """
    if len(image.shape) != 3:
        raise ValueError(f"Ожидается 3D массив для изображения, но получено {image.shape}")
    if len(cloud_mask.shape) != 2:
        raise ValueError(f"Ожидается 2D массив для маски облаков, но получено {cloud_mask.shape}")
        # Исправить размеры маски
    if cloud_mask.shape != image.shape[:2]:
        print(
            f"Размеры cloud_mask ({cloud_mask.shape}) и изображения ({image.shape[:2]}) не совпадают. Выполняется коррекция.")
        cloud_mask = resize(cloud_mask, image.shape[:2], preserve_range=True, anti_aliasing=False).astype(bool)
    if image.shape[:2] != cloud_mask.shape:
        raise ValueError(f"Размерность маски {cloud_mask.shape} не соответствует размерности изображения {image.shape[:2]}")

    # Создаем копию изображения для работы
    filled_image = np.copy(image)

    # Пройти по каждому каналу изображения
    for i in range(image.shape[2]):
        channel = image[:, :, i]
        mask = cloud_mask

        # Вычисляем расстояние до ближайшего не-облачного пикселя
        distance = distance_transform_edt(~mask)

        # Устанавливаем уровень размытия на основе среднего расстояния
        adaptive_sigma = np.mean(distance) / 10

        # Размываем значения, чтобы создать плавный переход
        blurred = gaussian_filter(channel, sigma=adaptive_sigma)

        # Заполняем облачные области размытыми значениями
        filled_image[:, :, i][mask] = blurred[mask]

    # Возвращаем изображение и метаданные
    metadata = {
        "geotransform": geotransform,
        "projection": projection
    }
    return filled_image, metadata


def extract_rgb(image, rgb_indices=(0, 1, 2)):
    """
    Извлечение RGB-изображения из многоканального изображения.

    :param image: ndarray, многоканальное изображение (H, W, C).
    :param rgb_indices: tuple, индексы каналов RGB (по умолчанию (0, 1, 2)).
    :return: ndarray, RGB-изображение (H, W, 3).
    """
    if image.shape[2] < max(rgb_indices) + 1:
        raise ValueError(f"Указанные RGB-индексы {rgb_indices} выходят за пределы количества каналов {image.shape[2]}.")

    # Извлечение RGB-каналов
    rgb_image = image[:, :, rgb_indices]

    # Нормализация значений в диапазон [0, 255] для визуализации
    rgb_image = (255 * (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))).astype(np.uint8)

    return rgb_image


# Комбинация изображений


# def remove_cloudss(dataset, bands, threshold=0.3):
#
#     # Читаем все слои (банды)
#     for i in range(1, dataset.RasterCount + 1):
#         band = dataset.GetRasterBand(i).ReadAsArray().astype(np.float32)
#         bands.append(band)
#
#     # Рассчитываем индекс яркости (например, среднее значение по спектральным диапазонам)
#     brightness = np.mean(bands, axis=0)
#
#     # Создаем маску облаков
#     cloud_mask = brightness > threshold
#
#     # Заменяем значения в облачных пикселях на NaN (или другое значение "нет данных")
#     result_bands = []
#     for band in bands:
#         band[cloud_mask] = np.nan  # Или другое значение, например, 0
#         result_bands.append(band)
#
#
# # Пример использования
# remove_clouds("input.tif", "output_no_clouds.tif")





# def save_tif(output_path, image, metadata):
#     """
#     Сохранение GeoTIFF с обработанным изображением.
#
#     :param output_path: str, путь для сохранения выходного GeoTIFF файла.
#     :param image: ndarray, 3D массив данных (каналы в последнем измерении).
#     :param metadata: dict, метаинформация изображения (геопривязка и проекция).
#     """
#     driver = gdal.GetDriverByName('GTiff')
#     rows, cols, bands = image.shape
#     output_dataset = driver.Create(output_path, cols, rows, bands, gdal.GDT_Float32)
#
#     # Устанавливаем метаинформацию
#     output_dataset.SetGeoTransform(metadata["geotransform"])
#     output_dataset.SetProjection(metadata["projection"])
#
#     # Сохраняем каждый канал
#     for i in range(bands):
#         output_dataset.GetRasterBand(i + 1).WriteArray(image[:, :, i])
#
#     output_dataset.FlushCache()
#     output_dataset = None  # Закрыть файл


# def save_multiband_image(channels, output_path, geo_transform, projection, no_data_value=np.nan):
#     """
#     Сохраняет многоканальное изображение в файл.
#
#     :param channels: Список каналов изображения (numpy.ndarray).
#     :param output_path: Путь к файлу для сохранения.
#     :param geo_transform: Геотрансформация GDAL.
#     :param projection: Проекция GDAL.
#     :param no_data_value: Значение "нет данных" для маски облаков.
#     """
#     driver = gdal.GetDriverByName('GTiff')
#     rows, cols = channels[0].shape
#     dataset = driver.Create(output_path, cols, rows, len(channels), gdal.GDT_Float32)
#
#     dataset.SetGeoTransform(geo_transform)
#     dataset.SetProjection(projection)
#
#     for i, channel in enumerate(channels, start=1):
#         band = dataset.GetRasterBand(i)
#         band.WriteArray(channel)
#         band.SetNoDataValue(no_data_value)
#
#     dataset = None
#     print(f"Файл сохранен: {output_path}")