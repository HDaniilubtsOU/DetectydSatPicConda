import numpy as np
from skimage import io, exposure, restoration, feature
from skimage.exposure import rescale_intensity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_image(image_path):
    """
    Загружает спутниковое изображение.
    :param image_path: Путь к GeoTIFF изображению.
    :return: NumPy массив изображения.
    """
    image = io.imread(image_path)
    print("Изображение загружено. Размерность:", image.shape)
    return image


def display_channels(image):
    """
    Отображает все каналы изображения.
    :param image: Многоканальное изображение (каналы, высота, ширина).
    """
    plt.figure(figsize=(12, 4))
    for i in range(image.shape[0]):
        plt.subplot(1, image.shape[0], i + 1)
        plt.imshow(image[i], cmap='gray')
        plt.title(f"Канал {i + 1}")
        plt.axis("off")
    plt.show()


def calculate_ndvi(image, nir_channel=3, red_channel=2):
    """
    Расчёт NDVI (индекса растительности).
    :param image: Многоканальное изображение.
    :param nir_channel: Индекс NIR канала.
    :param red_channel: Индекс Red канала.
    :return: NDVI как двумерный массив.
    """
    nir = image[nir_channel]
    red = image[red_channel]
    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi


def denoise_image(image, channel=2, weight=0.1):
    """
    Устраняет шум на указанном канале изображения.
    :param image: Многоканальное изображение.
    :param channel: Индекс канала для фильтрации.
    :param weight: Параметр фильтрации.
    :return: Фильтрованное изображение.
    """
    return restoration.denoise_tv_chambolle(image[channel], weight=weight)


def extract_texture_features(image, channel=2, distances=[5], angles=[0], levels=256):
    """
    Извлечение текстурных признаков (например, контрастность, энергия).
    :param image: Многоканальное изображение.
    :param channel: Индекс канала для анализа.
    :param distances: Расстояния для GLCM.
    :param angles: Углы для GLCM.
    :param levels: Количество уровней яркости.
    :return: Словарь с текстурными признаками.
    """
    image_8bit = (image[channel] / np.max(image[channel]) * 255).astype(np.uint8)
    glcm = feature.greycomatrix(image_8bit, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    features = {
        "contrast": feature.greycoprops(glcm, 'contrast')[0, 0],
        "energy": feature.greycoprops(glcm, 'energy')[0, 0],
    }
    return features


# _________________________________машинное обучение_____________________________________
def prepare_data(image, target):
    """
    Подготавливает данные для машинного обучения.
    :param image: Многоканальное изображение (каналы, высота, ширина).
    :param target: Целевая переменная (например, NDVI).
    :return: Признаки (X) и метки (y).
    """
    height, width = image.shape[1], image.shape[2]
    features = image.reshape(image.shape[0], -1).T
    target_flatten = target.flatten()

    # Удаление пикселей с отсутствием данных
    valid_mask = ~np.isnan(target_flatten)
    features = features[valid_mask]
    target = target_flatten[valid_mask]

    print("Данные подготовлены. Размерность признаков:", features.shape)
    return features, target


def train_model(X, y):
    """
    Обучает модель случайного леса.
    :param X: Признаки.
    :param y: Целевая переменная.
    :return: Обученная модель.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("R^2 на тестовом наборе:", model.score(X_test, y_test))
    return model


def predict_and_save(model, features, height, width, output_file="predicted_map.tif"):
    """
    Предсказывает и сохраняет результат как GeoTIFF.
    :param model: Обученная модель.
    :param features: Признаки для предсказания.
    :param height: Высота изображения.
    :param width: Ширина изображения.
    :param output_file: Путь к выходному файлу.
    """
    predicted = model.predict(features).reshape(height, width)
    plt.imshow(predicted, cmap='RdYlGn')
    plt.title("Предсказанная карта")
    plt.colorbar()
    plt.show()

    print("Карта предсказаний готова. Сохраните её в нужный формат.")


def normalize_skimage(image):
    """
    :return: Нормализованный массив
    """
    # Нормализация значений пикселей от 0 до 1
    normalized_image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    return normalized_image

