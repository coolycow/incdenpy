import math
import numpy as np

# Быстрые alias полей, которые используются в дальнейшей работе
xf = 'X'
yf = 'Y'
zf = 'Z'
tf = 'gps_time'
fl = 'point_source_id'

# Функция для квадрата расстояния 3D
def square_dist3(point_1, point_2):
    dx = point_1[xf] - point_2[xf]
    dy = point_1[yf] - point_2[yf]
    dz = point_1[zf] - point_2[zf]
    return dx * dx + dy * dy + dz * dz

# Функция для косинуса угла между направляющими векторами
def dir_cosines_z(point_1, point_2):
    dx = point_2[xf] - point_1[xf]
    dy = point_2[yf] - point_1[yf]
    dz = point_2[zf] - point_1[zf]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist == 0:
        return 0.0
    return dz / dist

# Функция сортировки точек с одинаковым временем
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
                points[i:j][xf].astype(np.float64),
                points[i:j][yf].astype(np.float64),
                points[i:j][zf].astype(np.float64)
            )).T

            check_coordinates = np.array([
                float(check_pnt[xf]),
                float(check_pnt[yf]),
                float(check_pnt[zf])
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

    # Максимальный угол переводим в радианы и вычитаем его из PI/2
    # Т.е. направляющий косинус у нас идет от начала вектора, то нам нужен обратный угол
    # Например, если у нас максимальный угол = 60 градусов, то реально нужно брать 30 и считать его как минимальный угол
    rad_angle = math.pi / 2 - max_angle * math.pi / 180

    # Выводим исходные классы
    logger.debug(f"Исходные классы: {np.unique(points[cf])}")

    # Сортируем имеющиеся точки по увеличению времени
    if points.size > 1:
        logger.debug("Быстрая сортировка точек по времени: начало")
        points.sort(order=tf)
        logger.debug("Быстрая сортировка точек по времени: конец")
    else:
        return points

    # Сортируем имеющиеся точки с одинаковым временем по расстоянию от точки с предыдущим временем
    if points.size > 1:
        logger.debug("Сортировка точек с одинаковым временем: начало")
        sort_with_same_time(points)
        logger.debug("Сортировка точек с одинаковым временем: конец")
    else:
        return points

    new_points = []

    n = len(points)

    # Шаблон для новой точки
    empty_point_template = np.empty(1, dtype=points.dtype)[0]

    for i in range(n - 1):
        pt1 = points[i]
        pt2 = points[i + 1]

        # Если у точек есть классы, то пропускаем точки, класс которых в ignore_set
        if has_cf and (pt1[cf] in ignore_set or pt2[cf] in ignore_set):
            continue

        # Если у точек есть включение, то пропускаем точки с разными номерами включений
        if has_fl and pt1[fl] != pt2[fl]:
            continue

        # Разница времени (GpsTime) между точками
        t1 = pt1[tf]
        t2 = pt2[tf]
        if abs(t2 - t1) > time_diff:
            continue

        # Квадрат расстояния 3D между точками
        qd = square_dist3(pt1, pt2)

        # Если по расстоянию точки слишком далеко - пропускаем вставку
        if qd > q_max_distance:
            continue

        # Угол между точками по Z направляющему косинусу
        angle = abs(math.acos(dir_cosines_z(pt1, pt2)))
        if angle < rad_angle and qd > q_angle_distance:
            # По углу и расстоянию не подходит - пропускаем вставку
            continue

        delta_time = abs(t2 - t1) / parts_count
        delta_dist_frac = 1 / parts_count

        for part in range(1, parts_count):
            frac = part * delta_dist_frac

            # Создаем новую точку на основе копии шаблона.
            # Переносим все поля из первой точки.
            # Нельзя делать interp_pt = np.copy(pt1), т.к. из-за особенностей копирования будет изменен исходная точка.
            interp_pt = empty_point_template.copy()
            for name in points.dtype.names:
                interp_pt[name] = pt1[name]

            # Линейная интерполяция координат
            interp_pt[xf] = pt1[xf] + (pt2[xf] - pt1[xf]) * frac
            interp_pt[yf] = pt1[yf] + (pt2[yf] - pt1[yf]) * frac
            interp_pt[zf] = pt1[zf] + (pt2[zf] - pt1[zf]) * frac

            # Линейная интерполяция времени
            interp_pt[tf] = t1 + delta_time * part

            # Класс точки
            if 0 <= point_class < 256:
                interp_pt[cf] = point_class
            else:
                interp_pt[cf] = pt1[cf]

            # Добавляем новую точку
            new_points.append(interp_pt)

    if new_points:
        # Объединяем исходные точки и новые
        result = np.concatenate([points, np.array(new_points, dtype=points.dtype)])
    else:
        result = points

    return result
