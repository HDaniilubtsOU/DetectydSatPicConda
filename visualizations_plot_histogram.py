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