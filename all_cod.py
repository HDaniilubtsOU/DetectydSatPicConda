from osgeo import gdal
from osgeo import ogr, osr, gdal_array, gdalconst
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from skimage import exposure


# ______________________________________digital_elevation_model.tif____________________________________________________
# dem_ds = gdal.Open('data/digital_elevation_model.tif', gdal.GA_ReadOnly)
#
# print('Basic metadata')
# print('Bands count: %s' % dem_ds.RasterCount)
# print('Data type: %s' % dem_ds.RasterCount)
# print('Rows: %s | Cols: %s' % (dem_ds.RasterYSize, dem_ds.RasterXSize))
# print('GeoTransform: %s' % str(dem_ds.GetGeoTransform()))
# print('Coordinate reference system: %s' % dem_ds.GetProjection())
#
#
#
# # Растр целиком
# dem_ds_data = dem_ds.ReadAsArray()
# # Данные конкретного канала
# dem_ds_band1 = dem_ds.GetRasterBand(1)
# dem_ds_band1_data = dem_ds_band1.ReadAsArray()
#
#
#
# print(dem_ds_data.shape)
# print(dem_ds_band1_data.shape)
# # значение высоты в ячейке 50, 50
# print (dem_ds_data[50,50])
# # срез значений высот
# print (dem_ds_data[40:48,40:48])
#
#
#
# # вычисление координат конкретной ячейки (50;50) на основе GeoTransform
# gt = dem_ds.GetGeoTransform()
# x_coord = gt[0] + 50*gt[1]
# y_coord = gt[3] + 50*gt[5]
# print (x_coord, y_coord)
#
#
#
# # средняя высота всего участка территории
# print (dem_ds_data.mean())
# # маска для ячеек со значением > 1500
# high_elevations = dem_ds_data[dem_ds_data > 1500]
# print (high_elevations.shape[0])
#
#
#
# # визуализация целиком
# plt.imshow(dem_ds_data)
# plt.colorbar()
# plt.show()



# # возвращаемся к набору dem_ds и перепроецируем его
# dem_ds_transformed = gdal.Warp('',dem_ds, format='MEM', dstSRS='EPSG:32634')
# """альтернативный вариант"""
# # dem_ds_transformed = gdal.Warp('data/transformed_dem.tif', dem_ds, format='GTiff', dstSRS='EPSG:32634')
# # del dem_ds_transformed
#
# # И всё готово, в переменной dem_ds_transformed хранящийся в памяти gdal.Dataset готовый к работе
# # читаем и смотрим на GeoTransform
# gt = dem_ds_transformed.GetGeoTransform()
# print('Новый GeoTransform:')
# print(gt)
# # читаем перепроецированную матрицу значений и применяем маску
# dem_ds_transformed_data = dem_ds_transformed.ReadAsArray()
# high_elevations = dem_ds_transformed_data[dem_ds_transformed_data > 1500]
#
# # Вычисляем площадь, умножая количество подходящих пикселей на площадь отдельного пикселя. Её вычисляем на основе данных Geotransform и приводим к км2
# total_area = high_elevations.shape[0] * (abs(gt[1])*abs(gt[5])) / 1000000
# print('Общая площадь районов с высотой > 1500 м: %s км2' % round(total_area,3))
#
# del dem_ds_transformed
# ______________________________________digital_elevation_model.tif____________________________________________________


# _____________________________________________multiband_imagery.tif___________________________________________________
# imagery_ds = gdal.Open('data/multiband_imagery.tif', gdal.GA_ReadOnly)
# # print ('Bands count: %s' % imagery_ds.RasterCount)
#
# imagery_ds_data = imagery_ds.ReadAsArray()
# # print (imagery_ds_data.shape)
#
#
#
# # # описываем функцию, которая будет нормализовать значения канала в диапазон от 0 до 1
# # def normalize(input_band):
# #     min_value, max_value = input_band.min()*1.0, input_band.max()*1.0
# #     return ((input_band*1.0 - min_value*1.0)/(max_value*1.0 - min_value))
# #
# #
# # # собираем матрицу из нормализованных каналов
# # rgb_normalized = np.dstack([normalize(imagery_ds_data[2]),
# #                             normalize(imagery_ds_data[1]),
# #                             normalize(imagery_ds_data[0])])
# #
# # plt.imshow(rgb_normalized)
# # plt.show()
#
#
#
# # Вычисляем NDVI
# ndvi = (imagery_ds_data[3] - imagery_ds_data[2]) / (imagery_ds_data[3] + imagery_ds_data[2])
#
# # # применяем жёлто-зелёную шкалу от 0 до 1. Чем зеленее, тем больше NDVI
# # plt.imshow(ndvi, cmap='YlGn', vmin=0, vmax=1)
# # plt.colorbar()
# # plt.show()
#
#
#
# # записываем матрицу в новый файл
# # Выбираем драйвер (формат выходного файла)
# driver = gdal.GetDriverByName("GTiff")
#
# # У нас будет 1 канал, только NDVI
# band_count = 1
#
# # Выбираем тип данных
# data_type = gdal.GDT_Float32
#
# # Создаём файл на основе выбранного драйвера, наследуя размер у исходного набора
# dataset = driver.Create('data/ndvi.tif', imagery_ds.RasterXSize, imagery_ds.RasterYSize, band_count, data_type)
#
# # Заимствуем GeoTransform и систему координат у исходного растра (новый такой же, т.к. произведен в его пространственном домене)
# dataset.SetProjection(imagery_ds.GetProjection())
# dataset.SetGeoTransform(imagery_ds.GetGeoTransform())
#
# # Записываем сами данные
# dataset.GetRasterBand(1).WriteArray(ndvi)
#
# del dataset
#
# # image = mpimg.imread('data/ndvi.tif')
# # plt.imshow(image)
# # plt.show()
# _____________________________________________multiband_imagery.tif___________________________________________________


# ______________________________________________land_cover_types.zip___________________________________________________
# # данные из архива (vsizip)
# ds_from_zip = gdal.Open('/vsizip/data/land_cover_types.zip/land_cover_types.tif')
# ds_from_zip_data = ds_from_zip.ReadAsArray()
# print ('Срез данных в растре из архива:')
# print (ds_from_zip_data[100:105, 100:105])
#
# # данные из сетевого хранилища (vsicurl)
# ds_from_url = gdal.Open('/vsicurl/https://demo.nextgis.com/api/resource/7220/cog')
# print ('\nРазмер сетевого растра:')
# print(ds_from_url.RasterXSize)
# print(ds_from_url.RasterYSize)
#
# # Читаем конкретные значения, не скачивая весь растр
# print('4 пикселя начиная с адреса 10000, 10000. Все каналы:')
# print(ds_from_url.ReadAsArray(10000, 10000, 2, 2))
# ______________________________________________land_cover_types.zip___________________________________________________


# __________________________________________________topomap.tif________________________________________________________
# # Посмотрим на исходные данные как на RGB картинку
# image = mpimg.imread('data/topomap.tif')
# plt.imshow(image)
# plt.show()
#
#
#
# # отправим их в gdal.Translate с двумя заданиями - конвертация формата и загрубление пространственного разрешения
# topomap_converted = gdal.Translate('data/topomap_converted.png', 'data/topomap.tif',
#                                    format='PNG', xRes=20, yRes=20, creationOptions=["WORLDFILE=YES"])
# del topomap_converted
#
# # Смотрим на новый растр
# image = mpimg.imread('data/topomap_converted.png')
# plt.imshow(image)
# plt.show()
# __________________________________________________topomap.tif________________________________________________________


# _________________________________________________pan-sharpened_______________________________________________________



'''brovey'''
# BandRed_out = BandRed_in / [(BandBlue_in + BandGreen_in + BandRed_in) * BandPan]

"""
'Gram-Schmidt'
GeoEye – 0.6, 0.85, 0.75, 0.3
IKONOS – 0.85, 0.65, 0.35, 0.9
QuickBird – 0.85, 0.7, 0.35, 1.0
WorldView-2 – 0.95, 0.7, 0.5, 1.0
"""

'''IHS (Intensity, Hue, Saturation)'''

'''HSV'''

'''Simple Mean'''

'''PCA (Principal Component Analysis)'''

# _________________________________________________pan-sharpened_______________________________________________________


# ___________________________________________________normalize_________________________________________________________
imagery_ds = gdal.Open('data/multiband_imagery.tif', gdal.GA_ReadOnly)
# print ('Bands count: %s' % imagery_ds.RasterCount)

imagery_ds_data = imagery_ds.ReadAsArray()
# print (imagery_ds_data.shape)

# Получаем количество каналов
bands = imagery_ds.RasterCount
print(f'Количество каналов: {bands}')


channels = []
# Получаем описание каждого канала
for i in range(1, bands + 1):
    # Предполагаем, что у вас 4 канала
    band = imagery_ds.GetRasterBand(i)
    channels.append(band.ReadAsArray())

    # data = band.ReadAsArray()
    # plt.imshow(data, cmap='gray')
    # plt.title(f'Канал {i}')
    # plt.colorbar()
    # plt.show()

# Например, RGB и NIR:
red, green, blue, nir = channels
# print(f"Минимум и максимум в красном канале: {red.min()}, {red.max()}")
# print(f"Минимум и максимум в зеленом канале: {green.min()}, {green.max()}")
# print(f"Минимум и максимум в голубом канале: {blue.min()}, {blue.max()}")
# print(f"Минимум и максимум в нир канале: {nir.min()}, {nir.max()}")

# Применение контрастного растяжения
red_stretched = exposure.equalize_hist(red)
green_stretched = exposure.equalize_hist(green)
blue_stretched = exposure.equalize_hist(blue)
nir_stretched = exposure.equalize_hist(nir)

# нормализация значений канала относительно глобального максимума
red_norm = np.clip(red / red.max(), 0, 1)
green_norm = np.clip(green / green.max(), 0, 1)
blue_norm = np.clip(blue / blue.max(), 0, 1)
nir_norm = np.clip(nir / nir.max(), 0, 1)
# print(f"Normalize для красного канала: {red_norm.min()}, {red_norm.max()}")
# print(f"Normalize для зеленого канала: {green_norm.min()}, {green_norm.max()}")
# print(f"Normalize для голубого канала: {blue_norm.min()}, {blue_norm.max()}")
# print(f"Normalize для нир канала: {nir_norm.min()}, {nir_norm.max()}")

# Применение контрастного растяжения
red_stretched_norm = np.clip(red_stretched / red_stretched.max(), 0, 1)
green_stretched_norm = np.clip(green_stretched / green_stretched.max(), 0, 1)
blue_stretched_norm = np.clip(blue_stretched / blue_stretched.max(), 0, 1)
nir_stretched_norm = np.clip(nir_stretched / nir_stretched.max(), 0, 1)

# plt.imshow(red, cmap='gray')
# plt.title(f'Канал Red')
# plt.colorbar()
# plt.show()


# описываем функцию, которая будет нормализовать значения канала в диапазон от 0 до 1 (линейная нормализация используется для более естественного результата)
def normalize(input_band):
    min_value, max_value = input_band.min() * 1.0, input_band.max() * 1.0
    return (input_band * 1.0 - min_value * 1.0) / (max_value * 1.0 - min_value)
# описываем функцию, которая будет нормализовать значения канала с удалением выбросов
def normalize2(input_band, clip_percentile=2):
    lower, upper = np.percentile(input_band, (clip_percentile, 100 - clip_percentile))
    clipped_band = np.clip(input_band, lower, upper)
    return (clipped_band - lower) / (upper - lower)
# Эта ^^^ версия обрезает выбросы (например, верхние и нижние 2% значений), что помогает сгладить сильные отклонения.


# собираем матрицу из нормализованных каналов
rgb_normalized = np.dstack([normalize(imagery_ds_data[2]),
                            normalize(imagery_ds_data[1]),
                            normalize(imagery_ds_data[0])])
# plt.imshow(rgb_normalized)
# plt.title("RGB 0-1")
# plt.axis('off')
# plt.show()

rgb_normalized_nir = np.dstack([normalize(imagery_ds_data[3])])

# собираем матрицу из нормализованных каналов
rgb_normalized2 = np.dstack([normalize2(imagery_ds_data[2]),
                            normalize2(imagery_ds_data[1]),
                            normalize2(imagery_ds_data[0])])
# plt.imshow(rgb_normalized2)
# plt.title("RGB с удалением выбросов")
# plt.axis('off')
# plt.show()

rgb_normalized_nir2 = np.dstack([normalize2(imagery_ds_data[3])])

rgb_image_normalized = np.dstack((red_norm, green_norm, blue_norm))
# plt.imshow(rgb_image)
# plt.title("RGB Глобальный максимум")
# plt.axis('off')
# plt.show()

# # Закрываем набор данных
# imagery_ds = None
# ___________________________________________________normalize_________________________________________________________


# _____________________________________________________NDVI____________________________________________________________
# Вычисляем NDVI без нормализации
ndvi_no_normalize = (imagery_ds_data[3] - imagery_ds_data[2]) / (imagery_ds_data[3] + imagery_ds_data[2])
# применяем жёлто-зелёную шкалу от 0 до 1. Чем зеленее, тем больше NDVI
plt.imshow(ndvi_no_normalize, cmap='YlGn', vmin=0, vmax=1)
plt.colorbar(label='NDVI')
plt.title("NDVI без нормализации")
plt.axis('off')
plt.show()
#
# Вычисляем NDVI c нормализацией
ndvi_with_normalize = (normalize(imagery_ds_data[3]) - normalize(imagery_ds_data[2])) / (normalize(imagery_ds_data[3]) + normalize(imagery_ds_data[2]))
# применяем жёлто-зелёную шкалу от 0 до 1. Чем зеленее, тем больше NDVI
# plt.imshow(ndvi, cmap='YlGn', vmin=0, vmax=1)
plt.imshow(ndvi_with_normalize, cmap='YlGn')
plt.colorbar(label='NDVI')
plt.title("NDVI c нормализации")
plt.axis('off')
plt.show()
# Вычисляем NDVI c нормализацией2
ndvi_with_normalize2 = (normalize2(imagery_ds_data[3]) - normalize2(imagery_ds_data[2])) / (normalize2(imagery_ds_data[3]) + normalize2(imagery_ds_data[2]))
# применяем жёлто-зелёную шкалу от 0 до 1. Чем зеленее, тем больше NDVI
plt.imshow(ndvi_with_normalize2, cmap='YlGn', vmin=0, vmax=1)
plt.colorbar(label='NDVI')
plt.title("NDVI c нормализации2")
plt.axis('off')
plt.show()

# ___лучшее___
# ЛУЧШЕ ВСЕГО ПОКАЗЫВАЕТ ВОДУ (по моему мнению)
ndvi_global_max = (nir_norm - red_norm) / (nir_norm + red_norm)
# Визуализация NDVI
plt.imshow(ndvi_global_max, cmap='RdYlGn')
plt.colorbar(label='NDVI')
plt.title("NDVI2")
plt.axis('off')
plt.show()

# Простая классификация NDVI
ndvi_class_global_max = np.where(ndvi_global_max < 0.2, 0, 1)  # 0 - низкая растительность, 1 - высокая
plt.imshow(ndvi_class_global_max, cmap='gray')
plt.title("Классификация NDVI")
plt.axis('off')
plt.show()
# ___лучшее___


ndvi_class = np.where(ndvi_no_normalize < 0.2, 0, 1)  # 0 - низкая растительность, 1 - высокая
plt.imshow(ndvi_class, cmap='gray')
plt.title("Классификация NDVI")
plt.axis('off')
plt.show()

ndvi_class = np.where(ndvi_with_normalize < 0.2, 0, 1)  # 0 - низкая растительность, 1 - высокая
plt.imshow(ndvi_class, cmap='gray')
plt.title("Классификация NDVI")
plt.axis('off')
plt.show()

ndvi_class = np.where(ndvi_with_normalize2 < 0.2, 0, 1)  # 0 - низкая растительность, 1 - высокая
plt.imshow(ndvi_class, cmap='gray')
plt.title("Классификация NDVI")
plt.axis('off')
plt.show()


# Закрываем набор данных
imagery_ds = None
# _____________________________________________________NDVI____________________________________________________________



"""Результаты"""


r_g_b_Nir, (re_d, gree_n, blu_e, ni_r) = plt.subplots(1, 4, figsize=(19, 4))
#
r_g_b_Nir_equalize, (red_equalize, green_equalize, blue_equalize, nir_equalize) = plt.subplots(1, 4, figsize=(19, 4))
# #
# r_g_b_Nir_normalize, (red_normalize, green_normalize,
#                       blue_normalize, nir_normalize) = plt.subplots(1, 4, figsize=(19, 4))
# #
# r_g_b_Nir_normalize_equalize, (red_normalize_equalize, green_normalize_equalize,
#                                blue_normalize_equalize, nir_normalize_equalize) = plt.subplots(1, 4, figsize=(19, 5))
# #
# rgb_rgbline_normalize, (rgb_norm, rgb_line_normalize, rgb_line_normalize_emissions)\
#     = plt.subplots(1, 3, figsize=(19, 4))
# #
# nir_line_normalize, (nir_max_normalize, nir_line_normalize1, nir_line_normalize2) = plt.subplots(1, 3, figsize=(18, 5))
# #
# rgb_ndvi_ndviClass, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
# # fig2, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(18, 6))


# ________________________________________Отображаем каналов R / G / B / Nir
re_d.imshow(red, cmap='Reds')
re_d.set_title("Red band")
re_d.axis('off')

gree_n.imshow(green, cmap='Greens')
gree_n.set_title("Green band")
gree_n.axis('off')

blu_e.imshow(blue, cmap='Blues')
blu_e.set_title("Blue band")
blu_e.axis('off')

ni_r.imshow(nir)
ni_r.set_title("Nir band")
ni_r.axis('off')

# ________________________________________Отображаем каналов R / G / B / Nir с контрастным растяжением
red_equalize.imshow(red_stretched, cmap='Reds')
red_equalize.set_title("Red band \n(с контрастным растяжением)")
red_equalize.axis('off')

green_equalize.imshow(green_stretched, cmap='Greens')
green_equalize.set_title("Green band \n(с контрастным растяжением)")
green_equalize.axis('off')

blue_equalize.imshow(blue_stretched, cmap='Blues')
blue_equalize.set_title("Blue band \n(с контрастным растяжением)")
blue_equalize.axis('off')

nir_equalize.imshow(nir_stretched)
nir_equalize.set_title("Nir band \n(с контрастным растяжением)")
nir_equalize.axis('off')


# # ________________________________________НОРМАЛИЗАЦИЯ относительно глобального максимума
# # ________________________________________Отображаем каналов R / G / B / Nir с нормализацией
# red_normalize.imshow(red_norm, cmap='Reds')
# red_normalize.set_title("Red band \nс норм глобал max")
# red_normalize.axis('off')
#
# green_normalize.imshow(green_norm, cmap='Greens')
# green_normalize.set_title("Green band \nс норм глобал max")
# green_normalize.axis('off')
#
# blue_normalize.imshow(blue_norm, cmap='Blues')
# blue_normalize.set_title("Blue band \nс норм глобал max")
# blue_normalize.axis('off')
#
# nir_normalize.imshow(nir_norm)
# nir_normalize.set_title("Nir band \nс норм глобал max")
# nir_normalize.axis('off')
#
# # ________________________________________Отображаем каналов R / G / B / Nir с нормализацией и контрастным растяжением
# red_normalize_equalize.imshow(red_stretched_norm, cmap='Reds')
# red_normalize_equalize.set_title("Red band \nс норм глобал max \nи\n контрастным растяжением")
# red_normalize_equalize.axis('off')
#
# green_normalize_equalize.imshow(green_stretched_norm, cmap='Greens')
# green_normalize_equalize.set_title("Green band \nс норм глобал max \nи\n контрастным растяжением")
# green_normalize_equalize.axis('off')
#
# blue_normalize_equalize.imshow(blue_stretched_norm, cmap='Blues')
# blue_normalize_equalize.set_title("Blue band \nс норм глобал max \nи\n контрастным растяжением")
# blue_normalize_equalize.axis('off')
#
# nir_normalize_equalize.imshow(nir_stretched_norm)
# nir_normalize_equalize.set_title("Nir band \nс норм глобал max \nи\n контрастным растяжением")
# nir_normalize_equalize.axis('off')


# # ________________________________________линейная НОРМАЛИЗАЦИЯ для более естественного результата
# # ____________________Отображаем RGB / RGB с линейная нормализацией / RGB с линейная нормализацией и удалением выбросов
# rgb_norm.imshow(rgb_image_normalized)
# rgb_norm.set_title("Rgb \nmax нормализацией")
# rgb_norm.axis('off')
#
# rgb_line_normalize.imshow(rgb_normalized)
# rgb_line_normalize.set_title("Rgb \nс линейной нормализацией")
# rgb_line_normalize.axis('off')
#
# rgb_line_normalize_emissions.imshow(rgb_normalized2)
# rgb_line_normalize_emissions.set_title("Rgb \nс линейной нормализацией и удалением выбросов")
# rgb_line_normalize_emissions.axis('off')
#
#
# # ____________________Отображаем nir с линейная нормализацией и удалением выбросов
# nir_max_normalize.imshow(nir_norm)
# nir_max_normalize.set_title("Nir \nс max нормализацией")
# nir_max_normalize.axis('off')
#
# nir_line_normalize1.imshow(rgb_normalized_nir)
# nir_line_normalize1.set_title("Nir \nс линейной нормализацией")
# nir_line_normalize1.axis('off')
#
# nir_line_normalize2.imshow(rgb_normalized_nir2)
# nir_line_normalize2.set_title("Nir \nс линейной нормализацией и удалением выбросов")
# nir_line_normalize2.axis('off')




# # ________________________________________Отображаем RGB / NDVI / NDVI_class
# ax1.imshow(rgb_image)
# ax1.set_title("RGB Глобальный максимум")
# ax1.axis('off')
#
# # Отображаем второе изображение
# ax2.imshow(ndvi_global_max, cmap='RdYlGn')
# ax2.set_title("NDVI2")
# ax2.axis('off')
#
# # Отображаем третье изображение
# ax3.imshow(ndvi_class_global_max, cmap='gray')
# ax3.set_title("Классификация NDVI")
# ax3.axis('off')


# Отображаем объединенное окно
plt.show()