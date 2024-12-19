import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(index_array, title="Index Histogram", xlabel="Value", ylabel="Frequency",
                   bins=50, color="blue", range=None, log_scale=False):
    """
    Строит информативную гистограмму для значений индекса.

    :param index_array: Массив значений индекса.
    :param title: Заголовок графика.
    :param xlabel: Подпись оси X.
    :param ylabel: Подпись оси Y.
    :param bins: Количество интервалов (по умолчанию 50).
    :param color: Цвет столбцов гистограммы.
    :param range: Кортеж (min, max) для диапазона значений по оси X.
    :param log_scale: Логарифмическая шкала для оси Y (по умолчанию False).
    """
    # Убираем NaN значения
    index_array = np.nan_to_num(index_array)

    plt.figure(figsize=(10, 6))
    plt.hist(index_array.flatten(), bins=bins, color=color, edgecolor="black", alpha=0.7, range=range)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log_scale:
        plt.yscale('log')  # Логарифмическая шкала для больших разбросов
    plt.grid(alpha=0.3)
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


# улучшенная версия
def plot_histograms_three_single_channel2(data1, data2, data3,
                                         titles=("Data 1", "Data 2", "Data 3"),
                                         bins=256, color='blue', alpha=0.6,
                                         log_scale=False):
    """
    Визуализирует гистограммы трёх данных (например, каналов или результатов) в одном окне.

    :param data1: Первый массив данных (например, канал 1).
    :param data2: Второй массив данных (например, канал 2).
    :param data3: Третий массив данных (например, канал 3).
    :param titles: Кортеж заголовков для каждого канала.
    :param bins: Количество интервалов для гистограммы.
    :param color: Цвет гистограмм.
    :param alpha: Прозрачность гистограмм.
    :param log_scale: Логарифмическая шкала для частот (по умолчанию False).
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))  # 3 строки, 1 столбец

    # Вспомогательная функция для построения гистограммы одного канала
    def plot_histogram(ax, data, title):
        ax.hist(data.ravel(), bins=bins, color=color, edgecolor='black', alpha=alpha)
        ax.set_title(title)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        if log_scale:
            ax.set_yscale('log')
        ax.grid(alpha=0.3)

    # Построение гистограмм для каждого канала
    plot_histogram(axs[0], data1, titles[0])
    plot_histogram(axs[1], data2, titles[1])
    plot_histogram(axs[2], data3, titles[2])

    # Автоматическое выравнивание графиков
    plt.tight_layout()
    plt.show()


# странные гистограммы и довольно таки бесполезные
def plot_index_and_channels_histogram(index_array, channels, channel_names,
                                      index_title="Index Histogram",
                                      bins=50, range=None, log_scale=False):
    """
    Строит гистограммы индекса и его каналов на одном графике.

    :param index_array: Массив значений индекса.
    :param channels: Список массивов каналов (например, [NIR, Red]).
    :param channel_names: Список названий каналов (например, ["NIR", "Red"]).
    :param index_title: Заголовок для индекса.
    :param bins: Количество интервалов (по умолчанию 50).
    :param range: Кортеж (min, max) для диапазона значений по оси X.
    :param log_scale: Логарифмическая шкала для оси Y (по умолчанию False).
    """
    num_channels = len(channels)

    # Создаем подграфики: один для индекса и остальные для каналов
    fig, axs = plt.subplots(1, num_channels + 1, figsize=(5 * (num_channels + 1), 5))

    # Гистограмма индекса
    axs[0].hist(index_array.flatten(), bins=bins, color="green", edgecolor="black", alpha=0.7, range=range)
    axs[0].set_title(index_title)
    axs[0].set_xlabel("Value")
    axs[0].set_ylabel("Frequency")
    if log_scale:
        axs[0].set_yscale('log')
    axs[0].grid(alpha=0.3)

    # Гистограммы каналов
    for i, (channel, name) in enumerate(zip(channels, channel_names), start=1):
        axs[i].hist(channel.flatten(), bins=bins, color="blue", edgecolor="black", alpha=0.7, range=range)
        axs[i].set_title(f"{name} Histogram")
        axs[i].set_xlabel("Value")
        axs[i].set_ylabel("Frequency")
        if log_scale:
            axs[i].set_yscale('log')
        axs[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_all_indices_histograms(indices_dict, channels_dict, bins=30, range=(-1, 1)):
    """
    Автоматически строит гистограммы всех индексов и их каналов.

    :param indices_dict: Словарь, где ключ - название индекса, значение - массив индекса.
    :param channels_dict: Словарь, где ключ - название индекса, значение - кортеж (список каналов, названия каналов).
    :param bins: Количество интервалов для гистограмм.
    :param range: Диапазон значений по оси X.
    """
    for index_name, index_array in indices_dict.items():
        channels, channel_names = channels_dict[index_name]
        plot_index_and_channels_histogram(
            index_array,
            channels,
            channel_names,
            index_title=f"{index_name} Histogram",
            bins=bins,
            range=range
        )






# # тест гистограмм (которые бесполезные)
    # arvi_normalize_band_global_max = arvi.calculate_arvi(
    #     types_normalize.normalize_band_global_max(NIR),
    #     types_normalize.normalize_band_global_max(Red),
    #     types_normalize.normalize_band_global_max(Blue)
    # )
    #
    # evi_normalize_band_global_max = evi.calculate_evi(
    #     types_normalize.normalize_band_global_max(NIR),
    #     types_normalize.normalize_band_global_max(Red),
    #     types_normalize.normalize_band_global_max(Blue)
    # )
    #
    # ndvi_normalize_band_global_max = ndvi.calculate_ndvi(types_normalize.normalize_band_global_max(NIR),
    #                                                      types_normalize.normalize_band_global_max(Red))
    #
    # ndwi_normalize_band_global_max = ndwi.calculate_ndwi(
    #     types_normalize.normalize_band_global_max(NIR),
    #     types_normalize.normalize_band_global_max(Green)
    # )
    #
    # savi_normalize_band_global_max = savi.calculate_savi(
    #     types_normalize.normalize_band_global_max(NIR),
    #     types_normalize.normalize_band_global_max(Red)
    # )
    #
    # # Генерация данных
    # indices_dict = {
    #     "NDVI": ndvi_normalize_band_global_max,
    #     "EVI": evi_normalize_band_global_max,
    #     "ARVI": arvi_normalize_band_global_max,
    #     "NDWI": ndwi_normalize_band_global_max,
    #     "SAVI": savi_normalize_band_global_max,
    # }
    #
    # channels_dict = {
    #     "NDVI": ([NIR, Red], ["NIR", "Red"]),
    #     "EVI": ([NIR, Red, Blue], ["NIR", "Red", "Blue"]),
    #     "ARVI": ([NIR, Red, Blue], ["NIR", "Red", "Blue"]),
    #     "NDWI": ([Green, NIR], ["Green", "NIR"]),
    #     "SAVI": ([NIR, Red], ["NIR", "Red"]),
    # }
    #
    # visualizations_plot_histogram.visualize_all_indices_histograms(indices_dict, channels_dict, bins=30, range=(-1, 1))