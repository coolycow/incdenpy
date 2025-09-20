import math
import multiprocessing
import numpy as np
from numba import njit
from concurrent.futures import ProcessPoolExecutor
from logger import get_worker_logger, init_worker_logger

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
        logger
):
    log = logger or get_worker_logger()
    log.debug(f"Обработка группы point_source_id={points_group[fl][0]}")

    if points_group.size <= 1:
        return points_group

    # Делаем оптимизацию входных данных
    ignore_set = set(ignore_classes)
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
    log.debug("Сортировка исходных точек по времени: начало")
    points_group.sort(order=tf)
    log.debug("Сортировка исходных точек по времени: конец")

    # Сортируем имеющиеся точки с одинаковым временем по расстоянию от точки с предыдущим временем
    log.debug("Сортировка исходных точек с одинаковым временем: начало")
    sort_with_same_time(points_group)
    log.debug("Сортировка исходных точек с одинаковым временем: конец")

    # Векторизированная фильтрация пар точек
    log.debug("Фильтрация исходного массива точек: начало")
    n = len(points_group)
    idx = np.arange(n - 1)

    mask_class = np.ones(n - 1, dtype=bool)
    if has_cf:
        mask_class = (
            ~np.isin(points_group[cf][idx], ignore_set) &
            ~np.isin(points_group[cf][idx + 1], ignore_set)
        )

    mask_fl = np.ones(n - 1, dtype=bool)
    if has_fl:
        mask_fl = points_group[fl][idx] == points_group[fl][idx + 1]

    mask_time = np.abs(points_group[tf][idx] - points_group[tf][idx + 1]) <= time_diff

    dx = points_group[xf][idx + 1] - points_group[xf][idx]
    dy = points_group[yf][idx + 1] - points_group[yf][idx]
    dz = points_group[zf][idx + 1] - points_group[zf][idx]
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
    log.debug("Фильтрация исходного массива точек: конец")

    # Предварительное вычисление долей для вставки
    delta_dist_frac = 1 / parts_count
    frac_values = np.arange(1, parts_count) * delta_dist_frac

    # Заготовки для вставки
    new_points = []
    empty_point_template = np.empty(1, dtype=points_group.dtype)[0]

    log.debug("Генерация новых точек: начало")
    for i in good_idx:
        pt1 = points_group[i]
        pt2 = points_group[i + 1]

        t1 = pt1[tf]
        t2 = pt2[tf]
        delta_time = abs(t2 - t1) / parts_count

        for part_idx, frac in enumerate(frac_values, start=1):
            interp_pt = empty_point_template.copy()
            for name in points_group.dtype.names:
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
    log.debug("Генерация новых точек: конец")

    if new_points:
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
        max_processes=4,
        test_mode=False,
        logger=None,
        process_log_dir="logs",
        process_log_filename = None
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

    # Разбиение точек по point_source_id
    # Если point_source_id нет, то формируем одну группу с полным массивом точек
    if fl not in points.dtype.names:
        logger.debug("Поля point_source_id нет: формируем одну группу с полным массивом точек")
        groups = [points]
    else:
        unique_fl = np.unique(points[fl])
        logger.debug(f"Всего уникальных point_source_id (fl): {len(unique_fl)}")
        groups = [points[points[fl] == val] for val in unique_fl]

    results = []
    max_workers = max(1, min(max_processes, multiprocessing.cpu_count()))
    logger.debug(f"Максимальное количество процессов: {max_workers}")

    if max_workers == 1:
        # Последовательная обработка без multiprocessing
        for group in groups:
            processed = process_group(group, scale, max_distance, max_angle, angle_distance,
                                    time_diff, quantity, point_class, ignore_classes, cf, logger)
            results.append(processed)
    else:
        # Параллельная обработка через ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_workers,
                                 initializer=init_worker_logger,
                                 initargs=(process_log_dir, process_log_filename)
                                 ) as executor:
            futures = []
            for group in groups:
                futures.append(executor.submit(process_group, group, scale, max_distance, max_angle, angle_distance,
                                               time_diff, quantity, point_class, ignore_classes, cf, logger))
            for future in futures:
                results.append(future.result())

    # Объединяем результаты
    final_points = np.concatenate(results)

    # Сортировка если нужно
    if final_sort:
        logger.debug("Сортировка итогового массива по времени")
        final_points.sort(order=tf)

    return final_points
