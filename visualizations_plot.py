import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np


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


def plot_index(index_data, title="Index Map", cmap='RdYlGn', vmin=-1, vmax=1, colorbar_label="Index Value"):
    """
    Визуализирует карту индекса с настраиваемой цветовой схемой.

    :param index_data: Массив данных индекса.
    :param title: Заголовок графика.
    :param cmap: Цветовая карта.
    :param vmin: Минимальное значение для цветовой шкалы.
    :param vmax: Максимальное значение для цветовой шкалы.
    :param colorbar_label: Название шкалы значений.
    """
    plt.imshow(index_data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.axis('off')
    plt.show()


# ARVI
def plot_arvi(arvi, title="ARVI Map"):
    plot_index(arvi, title=title, cmap='RdYlGn', vmin=-1, vmax=1, colorbar_label="ARVI Value")


# EVI
def plot_evi(evi, title="EVI Map"):
    plot_index(evi, title=title, cmap='YlGn', vmin=0, vmax=1, colorbar_label="EVI Value")
    # или plt.imshow(evi, cmap='viridis', vmin=-1, vmax=1)


# MNDWI
def plot_mndwi(mndwi, title="MNDWI Map"):
    plot_index(mndwi, title=title, cmap='Blues', vmin=-1, vmax=1, colorbar_label="MNDWI Value")


# NDVI
def plot_ndvi(ndvi, title="NDVI Map"):
    plot_index(ndvi, title=title, cmap='RdYlGn', vmin=-1, vmax=1, colorbar_label="NDVI Value")
    # plt.imshow(ndvi, cmap='jet', vmin=-1, vmax=1)          # крутой стиль, не нрав
    # plt.imshow(ndvi, cmap='BrBG', vmin=-1, vmax=1)         # тоже неплох


# NDWI
def plot_ndwi(ndwi, title="NDWI Map"):
    plot_index(ndwi, title=title, cmap='PuBu', vmin=-1, vmax=1, colorbar_label="NDWI Value")


# SAVI
def plot_savi(savi, title="SAVI Map"):
    plot_index(savi, title=title, cmap='YlGn', vmin=0, vmax=1, colorbar_label="SAVI Value")


# WRI
def plot_wri(wri, title="WRI Map"):
    plot_index(wri, title=title, cmap='GnBu', vmin=0, vmax=2, colorbar_label="WRI Value")


def plot_classification_index_with_labels(classified_index, labels, title="Index Classification", cmap_name="Set3", num_classes=4):
    """
    Визуализирует классифицированный индекс с метками для классов.

    :param classified_index: Классифицированный массив индекса.
    :param labels: Список строковых меток для каждого класса.
    :param title: Заголовок графика.
    :param cmap_name: Название цветовой карты.
    :param num_classes: Количество классов.
    """
    cmap = plt.get_cmap(cmap_name, num_classes)
    norm = BoundaryNorm(range(1, num_classes + 2), cmap.N)

    plt.imshow(classified_index, cmap=cmap, norm=norm)
    cbar = plt.colorbar(ticks=range(1, num_classes + 1))
    cbar.ax.set_yticklabels(labels)  # Устанавливаем текстовые метки для классов
    cbar.set_label("Classes")
    plt.title(title)
    plt.axis('off')
    plt.show()


# ARVI Classification
def plot_classification_arvi_with_labels(classified_arvi, title="ARVI Classification"):
    labels = ["Low Vegetation", "Moderate Vegetation", "High Vegetation", "Very High Vegetation"]
    plot_classification_index_with_labels(classified_arvi, labels, title=title, cmap_name="Set3", num_classes=4)


# EVI Classification
def plot_classification_evi_with_labels(classified_evi, title="EVI Classification"):
    labels = ["Low", "Moderate", "High", "Very High", "Extreme"]
    plot_classification_index_with_labels(classified_evi, labels, title=title, cmap_name="YlGn", num_classes=5)


# MNDWI Classification
def plot_classification_mndwi_with_labels(classified_mndwi, title="MNDWI Classification"):
    labels = ["Low Water", "Moderate Water", "High Water"]
    plot_classification_index_with_labels(classified_mndwi, labels, title=title, cmap_name="Blues", num_classes=3)


# NDVI Classification
def plot_classification_ndvi_with_labels(classified_ndvi, title="NDVI Classification"):
    labels = ["Barren Land", "Low Vegetation", "Moderate Vegetation", "High Vegetation", "Dense Vegetation"]
    plot_classification_index_with_labels(classified_ndvi, labels, title=title, cmap_name="RdYlGn", num_classes=5)


# NDWI Classification
def plot_classification_ndwi_with_labels(classified_ndwi, title="NDWI Classification"):
    labels = ["Low Water Content", "Moderate Water Content", "High Water Content"]
    plot_classification_index_with_labels(classified_ndwi, labels, title=title, cmap_name="PuBu", num_classes=3)


# SAVI Classification
def plot_classification_savi_with_labels(classified_savi, title="SAVI Classification"):
    labels = ["Low", "Moderate", "High", "Very High"]
    plot_classification_index_with_labels(classified_savi, labels, title=title, cmap_name="YlGn", num_classes=4)


# WRI Classification
def plot_classification_wri_with_labels(classified_wri, title="WRI Classification"):
    labels = ["Low Water Presence", "Moderate Water Presence", "High Water Presence"]
    plot_classification_index_with_labels(classified_wri, labels, title=title, cmap_name="GnBu", num_classes=3)


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


def plot_quantity_four_single_channel(channel1, channel2, channel3, channel4,
                                      titles=("Result 1", "Result 2", "Result 3", "Result 4"),
                                      colormap='gray'):
    """
    Визуализирует четыре отдельных изображения в одном окне.

    :param channel1: Первое изображение.
    :param channel2: Второе изображение.
    :param channel3: Третье изображение.
    :param channel4: Четвёртое изображение.
    :param titles: Заголовки для каждого изображения.
    :param colormap: Цветовая карта для отображения (по умолчанию 'gray').
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # 2 строки, 2 столбца

    # Первое изображение
    axs[0, 0].imshow(channel1, cmap=colormap)
    axs[0, 0].set_title(titles[0])
    axs[0, 0].axis('off')

    # Второе изображение
    axs[0, 1].imshow(channel2, cmap=colormap)
    axs[0, 1].set_title(titles[1])
    axs[0, 1].axis('off')

    # Третье изображение
    axs[1, 0].imshow(channel3, cmap=colormap)
    axs[1, 0].set_title(titles[2])
    axs[1, 0].axis('off')

    # Четвёртое изображение
    axs[1, 1].imshow(channel4, cmap=colormap)
    axs[1, 1].set_title(titles[3])
    axs[1, 1].axis('off')

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