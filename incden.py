import logging
import multiprocessing
import os
import threading
import traceback
from tqdm import tqdm
import laspy
import yaml
import argparse
from laspy import LaspyException, PackedPointRecord
from logger import setup_logger, log_config
from terrascan import increase_density_laspy
from concurrent.futures import ThreadPoolExecutor

# Загрузка конфигурации
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Парсинг аргументов командной строки
def parse_args(defaults):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config.yaml")
    parser.add_argument('--max_threads', type=int, default=defaults.get('max_threads'))
    parser.add_argument('--max_processes', type=int, default=defaults.get('max_processes'))
    parser.add_argument('--multithreading', action='store_true', default=defaults.get('multithreading'))
    parser.add_argument('--multiprocessing', action='store_true', default=defaults.get('multiprocessing'))
    parser.add_argument('--no_multithreading', action='store_false', dest='multithreading')
    parser.add_argument('--no_multiprocessing', action='store_false', dest='multiprocessing')
    parser.add_argument('--max_distance', type=float, default=defaults.get('max_distance'))
    parser.add_argument('--max_angle', type=float, default=defaults.get('max_angle'))
    parser.add_argument('--angle_distance', type=float, default=defaults.get('angle_distance'))
    parser.add_argument('--quantity', type=int, default=defaults.get('quantity'))
    parser.add_argument('--point_class', type=int, default=defaults.get('point_class'))
    parser.add_argument('--time_diff', type=float, default=defaults.get('time_diff'))
    parser.add_argument('--ignore_classes', type=int, nargs='*', default=defaults.get('ignore_classes'))
    parser.add_argument('--test', action='store_true', default=defaults.get('test'))
    parser.add_argument('--no_test', action='store_false', dest='test')
    parser.add_argument('--input_folder', type=str, default=defaults.get('input_folder'))
    parser.add_argument('--output_folder', type=str, default=defaults.get('output_folder'))
    parser.add_argument('--files', type=str, nargs='*', default=defaults.get('files'))
    parser.add_argument('--final_sort', action='store_true', default=defaults.get('final_sort'))
    parser.add_argument('--no_final_sort', action='store_false', dest='final_sort')
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
def process_files(process_settings):
    thread_logger = logging.getLogger(f"LaserDataLogger.{__name__}")

    if not process_settings['files']:
        directory = process_settings['input_folder']

        files = get_files_from_directory(directory)
        if not files:
            thread_logger.warning(f"В директории {directory} файлов не найдено")
            return
    else:
        files = set(process_settings['files'])

    if not process_settings['multithreading']:
        thread_logger.info(f"Начинается обработка {len(files)} файлов в одном потоке...")

        if process_settings['test'] and thread_logger:
            for filepath in files:
                process_file(filepath, process_settings)
        else:
            for filepath in tqdm(files, desc="Обработка файлов", unit="файл"):
                process_file(filepath, process_settings)
    else:
        max_threads = max(1, min(process_settings['max_threads'], multiprocessing.cpu_count()))
        thread_logger.info(f"Начинается обработка {len(files)} файлов с использованием {max_threads} потоков...")

        with ThreadPoolExecutor(max_workers=max_threads) as thread_executor:
            futures = {
                thread_executor.submit(process_file, filepath, process_settings):
                    filepath for filepath in files
            }

            # Используем tqdm только если test=False
            if process_settings['test'] and thread_logger:
                for future in futures:
                    filepath = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        thread_logger.error(f"Ошибка при обработке файла {filepath}: {e}")
            else:
                with tqdm(total=len(futures), desc="Обработка файлов", unit="файл") as pbar:
                    for future in futures:
                        filepath = futures[future]
                        try:
                            future.result()
                            pbar.update(1)
                        except Exception as e:
                            thread_logger.error(f"Ошибка при обработке файла {filepath}: {e}")
                            pbar.update(1)

# Обработка конкретного файла
def process_file(filepath, process_file_settings):
    file_logger = logging.getLogger(f"LaserDataLogger.{__name__}")

    if file_logger is None:
        raise RuntimeError("Logger не инициализирован")

    if not os.path.exists(filepath):
        file_logger.warning(f"Файл {filepath} не найден\n")
        return

    if process_file_settings['test'] and file_logger:
        file_logger.info(f"Начинается обработка файла: {filepath} (Поток: {threading.current_thread().name})")

    try:
        with laspy.open(filepath) as input_las:
            header = input_las.header

            # Выводим информацию о файле (формат точек, масштаб и смещение)
            if process_file_settings['test'] and file_logger:
                file_logger.debug(f"Format: {header.point_format}; Count: {header.point_count}; VLRs: {len(header.vlrs)}")
                file_logger.debug(f"Scale (units) X, Y, Z: {header.scale}")
                file_logger.debug(f"Offset X, Y, Z: {header.offset}")

            # Выводим все имена измерений
            if process_file_settings['test'] and file_logger:
                file_logger.debug("Point format dimension names:")
                for dim in header.point_format.dimension_names:
                    file_logger.debug(dim)

            # Читаем все точки из файла
            las = input_las.read()

            # Время может отсутствовать, учесть
            # Нельзя уплотнять точки без времени, так как невозможно гарантировать порядок точек
            try:
                _ = las.gps_time
            except AttributeError:
                file_logger.warning("В файле отсутствуют временные метки (gps_time). Уплотнение точек пропущено.")

                # Формируем имя для сохранения
                filename = os.path.basename(filepath)
                output_path = os.path.join(process_file_settings['output_folder'], filename)
                os.makedirs(process_file_settings['output_folder'], exist_ok=True)

                # Записываем исходный файл без изменений в выходную папку
                with open(output_path, "wb") as output_file:
                    las.write(output_file)

                file_logger.info(f"Исходный файл сохранён без изменений: {output_path}\n")
                return

            # Организуем данные в наш упрощенный класс Point3d для совместимости с increase_density
            points = las.points.array

            # Выводим первую точку для проверки координат
            if process_file_settings['test'] and file_logger:
                file_logger.debug(f"First point raw X, Y, Z: {points[0]['X']}, {points[0]['Y']}, {points[0]['Z']}")
                file_logger.debug(f"First point scaled X, Y, Z: {las.x[0]}, {las.y[0]}, {las.z[0]}")

            # Вызываем функцию на увеличение плотности точек с параметрами из process_file_settings
            new_points = increase_density_laspy(
                header.scale,
                points,
                max_distance = process_file_settings['max_distance'],
                max_angle = process_file_settings['max_angle'],
                angle_distance = process_file_settings['angle_distance'],
                time_diff = process_file_settings['time_diff'],
                quantity = process_file_settings['quantity'],
                point_class = process_file_settings['point_class'],
                ignore_classes = process_file_settings['ignore_classes'],
                final_sort = process_file_settings['final_sort'],
                use_multiprocessing = process_file_settings['multiprocessing'],
                max_processes = process_file_settings['max_processes'],
                test = process_file_settings['test']
            )

            # Проверяем, что действительно были добавлены новые точки
            if len(new_points) == len(points):
                if process_file_settings['test'] and file_logger:
                    file_logger.warning("Новых точек не добавлено, исходные данные остаются без изменений.")
            else:
                las.points = PackedPointRecord(new_points, header.point_format)
                las.header.point_count = len(new_points)
                if process_file_settings['test'] and file_logger:
                    file_logger.info(f"Добавлено дополнительных точек: {len(new_points) - len(points)}")

            # Формируем имя для сохранения
            filename = os.path.basename(filepath)
            output_path = os.path.join(process_file_settings['output_folder'], filename)

            # Обеспечиваем, что папка существует
            os.makedirs(process_file_settings['output_folder'], exist_ok=True)

            # Записываем обновленный las объект в новый файл
            with open(output_path, "wb") as output_file:
                las.write(output_file)
    except LaspyException as e:
        if process_file_settings['test'] and file_logger:
            file_logger.error(f"Ошибка чтения LAS/LAZ файла {filepath}: {e}. Файл пропущен\n")
        return
    except Exception as e:
        if file_logger:
            file_logger.error(f"Неизвестная ошибка при обработке файла {filepath}: {e}. Файл пропущен\n{traceback.format_exc()}")
        return

    if process_file_settings['test'] and file_logger:
        file_logger.info(f"Завершена обработка файла: {filepath}\n")

# Основная функция
if __name__ == "__main__":
    # Получаем только --config первым
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument('--config', default='config.yaml')
    args, remaining_argv = initial_parser.parse_known_args()

    config = load_config(args.config)
    settings = parse_args(config)

    # Настраиваем логирование
    main_logger = setup_logger()

    ascii_art_inc_den = r"""
  ####    ##   ##    ####   #####    #######  ##   ##
   ##     ###  ##   ##  ##   ## ##    ##   #  ###  ##
   ##     #### ##  ##        ##  ##   ## #    #### ##
   ##     ## ####  ##        ##  ##   ####    ## ####
   ##     ##  ###  ##        ##  ##   ## #    ##  ###
   ##     ##   ##   ##  ##   ## ##    ##   #  ##   ##
  ####    ##   ##    ####   #####    #######  ##   ##
        Welcome to LAS & LIDAR Processing Tool
       Increasing point cloud density with love
                         💙
"""

    print(ascii_art_inc_den)

    log_config(settings)

    # Запускаем обработку файлов
    process_files(settings)
