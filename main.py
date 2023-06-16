import os
import sys
import sqlite3
import datetime

from io import BytesIO
from time import sleep
# from config import FILENAME
from viewlet import DOG, WAVE, MATH
from libs import ImageManager, NormFactor, utils, SobelOperator


if __name__ == '__main__':
    SCALE = int(sys.argv[1])
    FILENAME = sys.argv[2]
    IMAGE = eval(sys.argv[3])
    print(SCALE, FILENAME, IMAGE)
    filename: str = FILENAME
    path_to_file = os.path.join(os.path.dirname(__file__), 'data.jpg')
    date = datetime.datetime.now()
    # print(config.SCALE)
    image, width, height, pixels, matrix = ImageManager.get_image_info(IMAGE)
    # Создание подключения к базе данных
    conn = sqlite3.connect('jupyter.sqlite3')
    # Создание курсора
    cursor = conn.cursor()
    # выполнение запроса на выборку метаданных таблиц (проверка на существование таблицы)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='image'")
    result = cursor.fetchone()
    # проверка наличия таблицы
    if not result:
        print("Таблица 'image' не существует")
        cursor.execute('''CREATE TABLE image
                (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                data BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
        cursor.execute('''CREATE TABLE processed_image
                (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                name_method TEXT NOT NULL,
                data BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(image_id) REFERENCES image(id)
                )
                ''')
        print(
            "БД image создана с полями (id(PK), name, data, created_at).",
            "БД processed_image создана с полями (id(PK), image_id(FK), name_method, data, created_at).",
            sep='\n'
        )

    # выполнение запроса на существование хотя бы одной записи в БД
    cursor.execute("SELECT count(*) FROM image")
    result = cursor.fetchone()
    # проверка количества записей
    # if result[0] > 0:
    #     # выполнение запроса на удаление всех записей
    #     cursor.execute("DELETE FROM image")
    #     conn.commit()
    #     print("Удалено записей:", cursor.rowcount)
    # else:
    #     print("Записей не обнаружено!")

    image_copy = image.copy()
    image_bytes = BytesIO()
    image_copy.save(image_bytes, format='JPEG')
    image_bytes_original = image_bytes.getvalue()
    cursor.execute(
        "INSERT INTO image (name, data, created_at) VALUES (?, ?, ?)",
        (filename, image_bytes_original, date)
    )

    fk_id = cursor.lastrowid

    print("DOG: ")
    image_dog = NormFactor.normalize_image(utils.grad(image, DOG.dwt_x(pixels, SCALE), DOG.dwt_y(pixels, SCALE)),
                                           matrix, width, height)
    image_bytes = BytesIO()
    image_dog.save(image_bytes, format='JPEG')
    image_bytes_dog = image_bytes.getvalue()
    image_dog.save('images/dog.png')
    # image_dog.show()

    cursor.execute(
        "INSERT INTO processed_image (image_id, name_method, data, created_at) VALUES (?, ?, ?, ?)",
        (fk_id, "dog", image_bytes_dog, date)
    )
    sleep(1)

    print("WAVE:")
    image_wave = NormFactor.normalize_image(utils.grad(image, WAVE.dwt_x(pixels, SCALE), WAVE.dwt_y(pixels, SCALE)),
                                            matrix, width, height)
    image_bytes = BytesIO()
    image_wave.save(image_bytes, format='JPEG')
    image_bytes_wave = image_bytes.getvalue()
    image_wave.save('images/wave.png')
    # image_wave.show()
    cursor.execute(
        "INSERT INTO processed_image (image_id, name_method, data, created_at) VALUES (?, ?, ?, ?)",
        (fk_id, "wave", image_bytes_wave, date)
    )
    sleep(1)

    print("MATH:")
    image_math = NormFactor.normalize_image(utils.grad(image, MATH.dx_math(pixels, SCALE), MATH.dy_math(pixels, SCALE)),
                                            matrix, width, height)
    image_bytes = BytesIO()
    image_math.save(image_bytes, format='JPEG')
    image_bytes_math = image_bytes.getvalue()
    image_math.save('images/math.png')
    # image_math.show()
    cursor.execute(
        "INSERT INTO processed_image (image_id, name_method, data, created_at) VALUES (?, ?, ?, ?)",
        (fk_id, "math", image_bytes_math, date)
    )
    sleep(1)

    print("Threshold DOG:")
    image_dog_threshold = utils.threshold(image_dog, width, height)
    image_bytes = BytesIO()
    image_dog_threshold.save(image_bytes, format='JPEG')
    image_bytes_dog_threshold = image_bytes.getvalue()
    image_dog_threshold.save('images/threshold_dog.png')
    # image_dog_threshold.show()
    cursor.execute(
        "INSERT INTO processed_image (image_id, name_method, data, created_at) VALUES (?, ?, ?, ?)",
        (fk_id, "threshold_dog", image_bytes_dog_threshold, date)
    )
    sleep(1)

    print("Threshold WAVE:")
    image_wave_threshold = utils.threshold(image_wave, width, height)
    image_bytes = BytesIO()
    image_wave_threshold.save(image_bytes, format='JPEG')
    image_bytes_wave_threshold = image_bytes.getvalue()
    image_wave_threshold.save('images/threshold_wave.png')
    # image_wave_threshold.show()
    cursor.execute(
        "INSERT INTO processed_image (image_id, name_method, data, created_at) VALUES (?, ?, ?, ?)",
        (fk_id, "threshold_wave", image_bytes_wave_threshold, date)
    )
    sleep(1)

    print("Threshold MATH:")
    image_math_threshold = utils.threshold(image_math, width, height)
    image_bytes = BytesIO()
    image_math_threshold.save(image_bytes, format='JPEG')
    image_bytes_math_threshold = image_bytes.getvalue()
    image_math_threshold.save('images/threshold_math.png')
    # image_math_threshold.show()
    cursor.execute(
        "INSERT INTO processed_image (image_id, name_method, data, created_at) VALUES (?, ?, ?, ?)",
        (fk_id, "threshold_math", image_bytes_math_threshold, date)
    )
    sleep(1)

    print("Sobel:")
    image_sobel = SobelOperator.sob_matrix(image, width, height)
    image_sobel_bytes = BytesIO()
    image_sobel.save(image_sobel_bytes, format='JPEG')
    image_sobel_bytes_bd = image_sobel_bytes.getvalue()
    image_sobel.save('images/sobel.png')
    cursor.execute(
        "INSERT INTO processed_image (image_id, name_method, data, created_at) VALUES (?, ?, ?, ?)",
        (fk_id, "sobel", image_sobel_bytes_bd, date)
    )
    sleep(1)

    conn.commit()
    conn.close()
