import math
import numpy as np
from numba import njit

# Быстрые alias полей, которые используются в дальнейшей работе
xf = 'X'
yf = 'Y'
zf = 'Z'
tf = 'gps_time'
fl = 'point_source_id'

# Функция сортировки точек с одинаковым временем
@njit
def sort_with_same_time(points):
    i = 0
    n = len(points)

    while i < n - 1:
        if points[tf][i] == points[tf][i + 1] and i != 0:
            # Контрольной точкой для расчета расстояний будет предыдущая точка списка
            check_pnt = points[i - 1]

            j = i
            while j < n and points[tf][i] == points[tf][j]:
                j += 1

            # Вычисляем расстояния до контрольной точки
            coordinates = np.vstack((
                points[i:j][xf],
                points[i:j][yf],
                points[i:j][zf]
            )).T

            check_coordinates = np.array([
                check_pnt[xf],
                check_pnt[yf],
                check_pnt[zf]
            ])

            dists = np.sum((coordinates - check_coordinates) ** 2, axis=1)
            order = np.argsort(dists)

            # Сортируем участок points по расстоянию
            points[i:j] = points[i:j][order]

            # Присваиваем i == j - 1, а дальше он еще раз увеличится на 1.
            # Делаем это потому что после break у нас j будет равна первой точке с другим временем
            i = j - 1
        i += 1

def increase_density_laspy(
        scale, # масштаб из header LAS-файла (например, Scale (units) X, Y, Z: [0.01 0.01 0.01])
        points, # numpy structured array или list dict-подобных с элементами
        max_distance, # максимальное расстояние (в системных единицах), при котором вставляем точки
        max_angle, # максимальный угол в градусах для ограничения вставки
        angle_distance, # порог расстояния для угловой проверки
        time_diff, # максимальная разница времени между соседними точками для вставки
        quantity, # количество новых точек, добавляемых между двумя соседними точками
        point_class, # класс для новых точек (-1 значит копировать класс исходной точки)
        ignore_classes, # список классов, которые игнорируются
        final_sort, # сортировка по времени для финальных точек
        test_mode=False, # зарезервировано для пометки точек
        logger=None
):
    # Проверяем, что вход - numpy array
    if not isinstance(points, np.ndarray):
        raise TypeError("points должен быть numpy.ndarray со структурированными типами")

    # Проверяем наличие полей XYZ
    for field in [xf, yf, zf]:
        if not field in points.dtype.names:
            raise TypeError("Массив должен содержать поля XYZ")

    # Проверяем наличие поля времени
    if not tf in points.dtype.names:
        raise TypeError("Массив должен содержать поле времени 'gps_time'")

    # Проверяем какая именно классификация используется (обычная или сырая)
    cf = 'classification'
    if not cf in points.dtype.names:
        cf = 'raw_classification'
        if test_mode and logger:
            logger.debug("Используется raw_classification")
    has_cf = (cf in points.dtype.names)

    # Проверяем наличие поля номера включения
    has_fl = (fl in points.dtype.names)

    # Делаем оптимизацию входных данных
    ignore_set = set(ignore_classes)

    # На всякий случай полученные значения переводим в положительные
    max_angle = abs(max_angle)

    # Возводим в квадрат полученные значения расстояний и переводим в целочисленные значения для ускорения расчетов
    # Если scale = 0.01, то unit = 100
    # scale[0] - это масштаб по оси X
    unit = 1 / scale[0]
    q_max_distance = (max_distance * unit) ** 2
    q_angle_distance = (angle_distance * unit) ** 2

    parts_count = abs(quantity) + 1

    # Максимальный угол переводим в радианы и вычитаем его из PI/2.
    # То есть направляющий косинус у нас идет от начала вектора, то нам нужен обратный угол
    # Например, если у нас максимальный угол = 60 градусов, то реально нужно брать 30 и считать его как минимальный угол
    rad_angle = math.pi / 2 - max_angle * math.pi / 180

    # Выводим исходные классы
    logger.debug(f"Исходные классы: {np.unique(points[cf])}")

    # Сортируем имеющиеся точки по увеличению времени
    if points.size > 1:
        logger.debug("Сортировка исходных точек по времени: начало")
        points.sort(order=tf)
        logger.debug("Сортировка исходных точек по времени: конец")
    else:
        return points

    # Сортируем имеющиеся точки с одинаковым временем по расстоянию от точки с предыдущим временем
    if points.size > 1:
        logger.debug("Сортировка исходных точек с одинаковым временем: начало")
        sort_with_same_time(points)
        logger.debug("Сортировка исходных точек с одинаковым временем: конец")
    else:
        return points

    n = len(points)

    # Шаблон для новой точки
    empty_point_template = np.empty(1, dtype=points.dtype)[0]

    logger.debug("Фильтрация исходного массива точек: начало")

    # Векторизированная фильтрация пар точек
    idx = np.arange(n - 1)

    mask_class = np.ones(n - 1, dtype=bool)
    if has_cf:
        mask_class = (
            ~np.isin(points[cf][idx], ignore_set) &
            ~np.isin(points[cf][idx + 1], ignore_set)
        )

    mask_fl = np.ones(n - 1, dtype=bool)
    if has_fl:
        mask_fl = points[fl][idx] == points[fl][idx + 1]

    mask_time = np.abs(points[tf][idx] - points[tf][idx + 1]) <= time_diff

    dx = points[xf][idx + 1] - points[xf][idx]
    dy = points[yf][idx + 1] - points[yf][idx]
    dz = points[zf][idx + 1] - points[zf][idx]
    qd = dx ** 2 + dy ** 2 + dz ** 2
    mask_dist = qd <= q_max_distance

    dist = np.sqrt(qd)
    cos_z = np.zeros(n - 1, dtype=np.float64)
    valid_dist = dist > 0
    cos_z[valid_dist] = dz[valid_dist] / dist[valid_dist]
    angle = np.abs(np.arccos(cos_z))
    mask_angle = ~((angle < rad_angle) & (qd > q_angle_distance))

    final_mask = mask_class & mask_fl & mask_time & mask_dist & mask_angle

    good_idx = np.where(final_mask)[0]

    # Предварительное вычисление долей для вставки
    delta_dist_frac = 1 / parts_count
    frac_values = np.arange(1, parts_count) * delta_dist_frac
    logger.debug("Фильтрация исходного массива точек: конец")

    logger.debug("Генерация новых точек: начало")
    new_points = []
    for i in good_idx:
        pt1 = points[i]
        pt2 = points[i + 1]

        t1 = pt1[tf]
        t2 = pt2[tf]
        delta_time = abs(t2 - t1) / parts_count

        for part_idx, frac in enumerate(frac_values, start=1):
            interp_pt = empty_point_template.copy()
            for name in points.dtype.names:
                interp_pt[name] = pt1[name]

            interp_pt[xf] = pt1[xf] + (pt2[xf] - pt1[xf]) * frac
            interp_pt[yf] = pt1[yf] + (pt2[yf] - pt1[yf]) * frac
            interp_pt[zf] = pt1[zf] + (pt2[zf] - pt1[zf]) * frac

            interp_pt[tf] = t1 + delta_time * part_idx

            if 0 <= point_class < 256:
                interp_pt[cf] = point_class
            else:
                interp_pt[cf] = pt1[cf]

            new_points.append(interp_pt)

    logger.debug("Генерация новых точек: конец")

    if new_points:
        # Объединяем исходные точки и новые
        result = np.concatenate([points, np.array(new_points, dtype=points.dtype)])

        # Сортируем точки по времени, чтобы результат был похож на исходный
        if final_sort:
            logger.debug("Сортировка финальных точек по времени: начало")
            result.sort(order=tf)
            logger.debug("Сортировка финальных точек по времени: конец")
    else:
        result = points

    return result
