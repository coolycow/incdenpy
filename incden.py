import multiprocessing
import os
import laspy
import yaml
import argparse
from laspy import LaspyException, PackedPointRecord
from logger import setup_logger, log_config, get_worker_logger, init_worker_logger
from terrascan import increase_density_laspy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# Загрузка конфигурации
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Парсинг аргументов командной строки
def parse_args(defaults):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config.yaml")
    parser.add_argument('--max_workers', type=int, default=defaults.get('max_workers'))
    parser.add_argument('--with_workers', action='store_true', default=defaults.get('with_workers'))
    parser.add_argument('--max_distance', type=float, default=defaults.get('max_distance'))
    parser.add_argument('--max_angle', type=float, default=defaults.get('max_angle'))
    parser.add_argument('--angle_distance', type=float, default=defaults.get('angle_distance'))
    parser.add_argument('--quantity', type=int, default=defaults.get('quantity'))
    parser.add_argument('--point_class', type=int, default=defaults.get('point_class'))
    parser.add_argument('--time_diff', type=float, default=defaults.get('time_diff'))
    parser.add_argument('--ignore_classes', type=int, nargs='*', default=defaults.get('ignore_classes'))
    parser.add_argument('--test_mode', action='store_true', default=defaults.get('test_mode'))
    parser.add_argument('--input_folder', type=str, default=defaults.get('input_folder'))
    parser.add_argument('--output_folder', type=str, default=defaults.get('output_folder'))
    parser.add_argument('--save_format', type=str, choices=['', 'las10', 'las11', 'las12', 'las13', 'las14'], default=defaults.get('save_format'))
    return vars(parser.parse_args())

# Получение списка файлов из директории
def get_files_from_directory(directory):
    """Возвращает список путей к файлам в директории."""
    files = []
    for entry in os.scandir(directory):
        if entry.is_file():
            files.append(entry.path)
    return files

# Обработка файлов в директории
def process_files(process_settings, process_logger, process_log_dir="logs", process_log_filename = None):
    directory = process_settings['input_folder']

    files = get_files_from_directory(directory)
    if not files:
        process_logger.warning(f"В директории {directory} файлов не найдено")
        return

    if not process_settings['with_workers']:
        process_logger.info(f"Начинается обработка {len(files)} файлов в одном потоке...")
        for filepath in files:
            process_file(filepath, process_settings, process_logger)
    else:
        max_workers = max(1, min(process_settings['max_workers'], multiprocessing.cpu_count()))
        process_logger.info(f"Начинается обработка {len(files)} файлов с использованием {max_workers} потоков...")

        with ProcessPoolExecutor(max_workers=max_workers,
                                 initializer=init_worker_logger,
                                 initargs=(process_log_dir, process_log_filename)) as executor:
            futures = {
                executor.submit(process_file, filepath, process_settings, process_logger):
                    filepath for filepath in files
            }

            for future in as_completed(futures):
                filepath = futures[future]
                try:
                    future.result()
                except Exception as e:
                    process_logger.error(f"Ошибка при обработке файла {filepath}: {e}")

# Обработка конкретного файла
def process_file(filepath, process_file_settings, process_file_logger):
    log = process_file_logger or get_worker_logger()

    if log is None:
        raise RuntimeError("Logger не инициализирован")

    if not os.path.exists(filepath):
        log.warning(f"Файл {filepath} не найден\n")
        return

    log.info(f"Начинается обработка файла: {filepath}")

    try:
        with laspy.open(filepath) as input_las:
            # Выводим информацию о файле (формат точек, масштаб и смещение)
            header = input_las.header
            log.info(f"Format: {header.point_format}; Count: {header.point_count}; VLRs: {len(header.vlrs)}")
            log.info(f"Scale (units) X, Y, Z: {header.scale}")
            log.info(f"Offset X, Y, Z: {header.offset}")

            # Выводим все имена измерений
            if process_file_settings['test_mode']:
                log.debug("Point format dimension names:")
                for dim in header.point_format.dimension_names:
                    log.debug(dim)

            las = input_las.read()

            # Время может отсутствовать, учесть
            # Нельзя уплотнять точки без времени, так как невозможно гарантировать порядок точек
            try:
                _ = las.gps_time
            except AttributeError:
                log.warning("В файле отсутствуют временные метки (gps_time). Уплотнение точек пропущено.")

                # Формируем имя для сохранения
                filename = os.path.basename(filepath)
                output_path = os.path.join(process_file_settings['output_folder'], filename)
                os.makedirs(process_file_settings['output_folder'], exist_ok=True)

                # Записываем исходный файл без изменений в выходную папку
                with open(output_path, "wb") as output_file:
                    las.write(output_file)

                log.info(f"Исходный файл сохранён без изменений: {output_path}\n")
                return

            # Организуем данные в наш упрощенный класс Point3d для совместимости с increase_density
            points = las.points.array

            # Выводим первую точку для проверки координат
            log.info(f"First point raw X, Y, Z: {points[0]['X']}, {points[0]['Y']}, {points[0]['Z']}")
            log.info(f"First point scaled X, Y, Z: {las.x[0]}, {las.y[0]}, {las.z[0]}")

            # Вызываем функцию на увеличение плотности точек с параметрами из process_file_settings
            new_points = increase_density_laspy(
                header.scale,
                points,
                max_distance=process_file_settings['max_distance'],
                max_angle=process_file_settings['max_angle'],
                angle_distance=process_file_settings['angle_distance'],
                time_diff=process_file_settings['time_diff'],
                quantity=process_file_settings['quantity'],
                point_class=process_file_settings['point_class'],
                ignore_classes=process_file_settings['ignore_classes'],
                test_mode=process_file_settings['test_mode'],
                logger=log
            )

            if len(new_points) == len(points):
                log.info("Новых точек не добавлено, исходные данные остаются без изменений.")
            else:
                las.points = PackedPointRecord(new_points, header.point_format)
                las.header.point_count = len(new_points)
                log.info(f"Добавлено дополнительных точек: {len(new_points) - len(points)}")

            # Формируем имя для сохранения
            filename = os.path.basename(filepath)
            output_path = os.path.join(process_file_settings['output_folder'], filename)

            # Обеспечиваем, что папка существует
            os.makedirs(process_file_settings['output_folder'], exist_ok=True)

            # Записываем обновленный las объект в новый файл
            with open(output_path, "wb") as output_file:
                las.write(output_file)
    except LaspyException as e:
        log.error(f"Ошибка чтения LAS/LAZ файла: {e}. Файл пропущен\n")
        return
    except Exception as e:
        log.error(f"Неизвестная ошибка при обработке файла: {e}. Файл пропущен\n")
        return

    log.info(f"Завершена обработка файла: {filepath}\n")

# Основная функция
if __name__ == "__main__":
    # Получаем только --config первым
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument('--config', default='config.yaml')
    args, remaining_argv = initial_parser.parse_known_args()

    config = load_config(args.config)
    settings = parse_args(config)

    # Настраиваем логирование
    log_dir = "logs"
    log_filename = datetime.now().strftime("run_%Y%m%d_%H%M%S.log")
    logger, log_file = setup_logger(log_dir, log_filename)
    log_config(logger, settings)

    # Запускаем обработку файлов
    process_files(settings, logger, log_dir, log_filename)
