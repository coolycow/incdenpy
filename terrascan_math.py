import math
from point3d import Point3d

class TerraScanMath:
    # Метод для возведения в квадрат расстояния между двумя 3D точками (целочисленные координаты)
    @staticmethod
    def q_distance3(a: Point3d, b: Point3d) -> float:
        dx = a.x - b.x
        dy = a.y - b.y
        dz = a.z - b.z
        return dx * dx + dy * dy + dz * dz

    # Конвертация градусов в радианы
    @staticmethod
    def deg_to_rad(a: float) -> float:
        return a * math.pi / 180

    # Получение направляющих косинусов между двумя точками
    @staticmethod
    def direction_cosines(a: Point3d, b: Point3d) -> Point3d:
        ab_x = b.x - a.x
        ab_y = b.y - a.y
        ab_z = b.z - a.z
        length = math.sqrt(ab_x * ab_x + ab_y * ab_y + ab_z * ab_z)
        if length == 0:
            return Point3d(0, 0, 0)
        return Point3d(ab_x / length, ab_y / length, ab_z / length)
