import logging
from logger import setup_logger
import math
import multiprocessing
import numpy as np
from numba import njit
from concurrent.futures import ProcessPoolExecutor

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
            check_pnt = points[i - 1]

            j = i
            while j < n and points[tf][i] == points[tf][j]:
                j += 1

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

            points[i:j] = points[i:j][order]
            i = j - 1
        i += 1

# Обработка группы точек, группа - это точки одного включения
def process_group(
        points_group,
        scale,
        max_distance,
        max_angle,
        angle_distance,
        time_diff,
        quantity,
        point_class,
        ignore_classes,
        cf,
        test=False
):
    log = None

    if test:
        setup_logger()
        log = logging.getLogger(f"LaserDataLogger.{__name__}")

    if test and log:
        log.debug(f"Обработка группы point_source_id={points_group[fl][0]}")

    if points_group.size <= 1:
        return points_group

    # Делаем оптимизацию входных данных
    ignore_list = list(ignore_classes)
    max_angle_local = abs(max_angle)

    # Возводим в квадрат полученные значения расстояний и переводим в целочисленные значения для ускорения расчетов.
    # Если scale = 0.01, то unit = 100.
    # scale[0] - это масштаб по оси X.
    unit = 1 / scale[0]
    q_max_distance = (max_distance * unit) ** 2
    q_angle_distance = (angle_distance * unit) ** 2

    parts_count = abs(quantity) + 1

    # Максимальный угол переводим в радианы и вычитаем его из PI/2.
    # То есть направляющий косинус у нас идет от начала вектора, то нам нужен обратный угол.
    # Например, если у нас максимальный угол = 60 градусов, то реально нужно брать 30 и считать его как минимальный угол.
    rad_angle = math.pi / 2 - max_angle_local * math.pi / 180

    # Определение наличия полей классификации и включений.
    has_cf = (cf in points_group.dtype.names)
    has_fl = (fl in points_group.dtype.names)

    # Сортируем имеющиеся точки по увеличению времени
    if test and log:
        log.debug("Сортировка исходных точек по времени: начало")
    points_group.sort(order=tf)
    if test and log:
        log.debug("Сортировка исходных точек по времени: конец")

    # Сортируем имеющиеся точки с одинаковым временем по расстоянию от точки с предыдущим временем
    if test and log:
        log.debug("Сортировка исходных точек с одинаковым временем: начало")
    sort_with_same_time(points_group)
    if test and log:
        log.debug("Сортировка исходных точек с одинаковым временем: конец")

    # Векторизированная фильтрация пар точек
    if test and log:
        log.debug("Фильтрация исходного массива точек: начало")
    n = len(points_group)
    idx = np.arange(n - 1)

    mask_class = np.ones(n - 1, dtype=bool)
    if has_cf:
        mask_class = (
            ~np.isin(points_group[cf][idx], ignore_list) &
            ~np.isin(points_group[cf][idx + 1], ignore_list)
        )

    mask_fl = np.ones(n - 1, dtype=bool)
    if has_fl:
        mask_fl = points_group[fl][idx] == points_group[fl][idx + 1]

    mask_time = np.abs(points_group[tf][idx] - points_group[tf][idx + 1]) <= time_diff

    dx = points_group[xf][idx + 1].astype(np.float64) - points_group[xf][idx].astype(np.float64)
    dy = points_group[yf][idx + 1].astype(np.float64) - points_group[yf][idx].astype(np.float64)
    dz = points_group[zf][idx + 1].astype(np.float64) - points_group[zf][idx].astype(np.float64)
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
    if test and log:
        log.debug("Фильтрация исходного массива точек: конец")

    # Предварительное вычисление долей для вставки
    delta_dist_frac = 1 / parts_count
    frac_values = np.arange(1, parts_count) * delta_dist_frac

    if test and log:
        log.debug("Генерация новых точек: начало")

    # Получаем индексы "хороших" пар
    good_pairs = points_group[good_idx]
    next_points = points_group[good_idx + 1]

    # Извлекаем координаты и время
    x1 = good_pairs[xf]
    y1 = good_pairs[yf]
    z1 = good_pairs[zf]
    t1 = good_pairs[tf]

    x2 = next_points[xf]
    y2 = next_points[yf]
    z2 = next_points[zf]
    t2 = next_points[tf]

    # Создаём матрицу коэффициентов интерполяции
    frac_matrix = np.tile(frac_values, len(good_idx)).reshape(len(good_idx), -1)

    # Интерполируем координаты и время
    new_x = x1[:, None] + (x2[:, None] - x1[:, None]) * frac_matrix
    new_y = y1[:, None] + (y2[:, None] - y1[:, None]) * frac_matrix
    new_z = z1[:, None] + (z2[:, None] - z1[:, None]) * frac_matrix
    new_t = t1[:, None] + (t2[:, None] - t1[:, None]) * frac_values[None, :]

    # Создаём шаблон новой точки
    dtype = points_group.dtype
    new_point_shape = (len(good_idx) * len(frac_values),)
    new_points = np.zeros(new_point_shape, dtype=dtype)

    # Заполняем координаты и время
    new_points[xf] = new_x.flatten()
    new_points[yf] = new_y.flatten()
    new_points[zf] = new_z.flatten()
    new_points[tf] = new_t.flatten()

    # Копируем остальные поля из pt1
    for field in dtype.names:
        if field not in [xf, yf, zf, tf]:
            src = good_pairs[field][:, None]
            new_points[field] = src.repeat(len(frac_values), axis=0).flatten()

    # Устанавливаем классификацию
    if 0 <= point_class < 256:
        new_points[cf] = point_class
    else:
        new_points[cf] = good_pairs[cf][:, None].repeat(len(frac_values), axis=0).flatten()

    if test and log:
        log.debug("Генерация новых точек: конец")

    if new_points.size > 0:
        result = np.concatenate([points_group, np.array(new_points, dtype=points_group.dtype)])
    else:
        result = points_group

    return result

def increase_density_laspy(
        scale,
        points,
        max_distance,
        max_angle,
        angle_distance,
        time_diff,
        quantity,
        point_class,
        ignore_classes,
        final_sort,
        use_multiprocessing=True,
        max_processes=4,
        test=False
):
    logger = logging.getLogger(f"LaserDataLogger.{__name__}")

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
        if test and logger:
            logger.debug("Используется raw_classification")

    # Разбиение точек по point_source_id
    # Если point_source_id нет, то формируем одну группу с полным массивом точек
    if fl not in points.dtype.names:
        if test and logger:
            logger.debug("Поля point_source_id нет: формируем одну группу с полным массивом точек")
        groups = [points]
    else:
        unique_fl = np.unique(points[fl])
        if test and logger:
            logger.debug(f"Всего уникальных point_source_id (fl): {len(unique_fl)}")
        groups = [points[points[fl] == val] for val in unique_fl]

    if use_multiprocessing:
        max_workers = max(1, min(max_processes, multiprocessing.cpu_count()))
    else:
        max_workers = 1

    if test and logger:
        logger.debug(f"Максимальное количество процессов: {max_workers}")

    results = []

    if max_workers == 1:
        for group in groups:
            processed = process_group(group, scale, max_distance, max_angle, angle_distance,
                                    time_diff, quantity, point_class, ignore_classes, cf, test)
            results.append(processed)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for group in groups:
                futures.append(executor.submit(process_group, group, scale, max_distance, max_angle, angle_distance,
                                               time_diff, quantity, point_class, ignore_classes, cf, test))
            for future in futures:
                results.append(future.result())

    # Объединяем результаты
    final_points = np.concatenate(results)

    # Сортировка если нужно
    if final_sort:
        if test and logger:
            logger.debug("Сортировка итогового массива по времени")
        final_points.sort(order=tf)

    return final_points
