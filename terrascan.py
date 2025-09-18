import math
import numpy as np

def increase_density_laspy(
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

    # Быстрые alias полей, которые используются в дальнейшей работе
    xf = 'X'
    yf = 'Y'
    zf = 'Z'
    tf = 'gps_time'
    fl = 'point_source_id'
    cf = 'classification'

    # Проверяем наличие полей XYZ
    for field in [xf, yf, zf]:
        if not field in points.dtype.names:
            raise TypeError("Массив должен содержать поля XYZ")

    # Проверяем наличие поля времени
    if not tf in points.dtype.names:
        raise TypeError("Массив должен содержать поле времени 'gps_time'")

    # Проверяем какая именно классификация используется (обычная или сырая)
    if not cf in points.dtype.names:
        cf = 'raw_classification'
        if test_mode and logger:
            logger.debug("Используется raw_classification")
    has_cf = (cf in points.dtype.names)

    # Проверяем наличие поля номера включения
    has_fl = (fl in points.dtype.names)

    # Делаем оптимизацию входных данных
    ignore_set = set(ignore_classes)
    max_distance = abs(max_distance)
    max_angle = abs(max_angle)
    angle_distance = abs(angle_distance)
    q_angle_distance = angle_distance * angle_distance
    q_max_distance = max_distance * max_distance

    parts_count = quantity + 1
    rad_angle = math.pi / 2 - max_angle * math.pi / 180  # перевод max_angle градусов в радианы и расчет

    new_points = []

    # Функция для квадрата расстояния 3D
    def square_dist3(point_1, point_2):
        dx = float(point_1[xf]) - float(point_2[xf])
        dy = float(point_1[yf]) - float(point_2[yf])
        dz = float(point_1[zf]) - float(point_2[zf])
        return dx * dx + dy * dy + dz * dz

    # Функция для косинуса угла между направляющими векторами
    def dir_cosines_z(point_1, point_2):
        dx = float(point_2[xf]) - float(point_1[xf])
        dy = float(point_2[yf]) - float(point_1[yf])
        dz = float(point_2[zf]) - float(point_1[zf])
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist == 0:
            return 0.0
        return dz / dist

    n = len(points)

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
            interp_pt = np.copy(pt1)  # Копируем структуру точки

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
