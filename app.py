import os
import sys
import sqlite3
import datetime

from io import BytesIO
from time import sleep

import cv2
import numpy as np
import pytesseract
from roboflow import Roboflow

from viewlet import DOG, WAVE, MATH
from libs import NormFactor, utils, SobelOperator

from PIL import Image
from PyQt5 import uic, Qt, QtCore
from PyQt5 import QtGui
from PyQt5.Qt import QObject
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSignal, QThread, QTime, QTimer
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QLabel, QHeaderView, QTableWidgetItem

from libs import MassNoize, ImageManager
from libs.utils import FileDialog, sko, ImgCrop
from ui import design

from qt_material import apply_stylesheet, QtStyleTools


class ScriptThread(QThread):
    data = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.scale_value = 0
        self.image = None
        self.image_path = None

    def set_value(self, param, image, image_path):
        self.scale_value = param
        self.image = image
        self.image_path = image_path

    def run(self):
        image_array = []
        SCALE = int(self.scale_value)
        FILENAME = self.image_path
        IMAGE = self.image
        filename: str = FILENAME
        SCALE = 3

        print(SCALE, FILENAME, IMAGE)

        date = datetime.datetime.now()
        image, width, height, pixels, matrix = ImageManager.get_image_info2(IMAGE)

        image_copy = image.copy()
        image_bytes = BytesIO()
        image_copy.save(image_bytes, format='JPEG')
        image_bytes_original = image_bytes.getvalue()

        image_array.append(image_bytes_original)

        print("DOG: ")
        image_dog = NormFactor.normalize_image(
            utils.grad(image, DOG.dwt_x(pixels, SCALE), DOG.dwt_y(pixels, SCALE)),
            matrix, width, height)
        image_bytes = BytesIO()
        image_dog.save(image_bytes, format='JPEG')
        image_bytes_dog = image_bytes.getvalue()
        image_dog.save('images/dog.png')
        # image_dog.show()

        dog_sko = sko(image, image_dog)

        image_array.append(image_bytes_dog)

        print("WAVE:")
        image_wave = NormFactor.normalize_image(
            utils.grad(image, WAVE.dwt_x(pixels, SCALE), WAVE.dwt_y(pixels, SCALE)),
            matrix, width, height)
        image_bytes = BytesIO()
        image_wave.save(image_bytes, format='JPEG')
        image_bytes_wave = image_bytes.getvalue()
        image_wave.save('images/wave.png')

        wave_sko = sko(image, image_wave)

        image_array.append(image_bytes_wave)

        print("MATH:")
        image_math = NormFactor.normalize_image(
            utils.grad(image, MATH.dx_math(pixels, SCALE), MATH.dy_math(pixels, SCALE)),
            matrix, width, height)
        image_bytes = BytesIO()
        image_math.save(image_bytes, format='JPEG')
        image_bytes_math = image_bytes.getvalue()
        image_math.save('images/math.png')

        math_sko = sko(image, image_math)

        image_array.append(image_bytes_math)

        print("Threshold DOG:")
        image_dog_threshold = utils.threshold(image_dog, width, height)
        image_bytes = BytesIO()
        image_dog_threshold.save(image_bytes, format='JPEG')
        image_bytes_dog_threshold = image_bytes.getvalue()
        image_dog_threshold.save('images/threshold_dog.png')

        dog_threshold_sko = sko(image, image_dog_threshold)

        image_array.append(image_bytes_dog_threshold)

        print("Threshold WAVE:")
        image_wave_threshold = utils.threshold(image_wave, width, height)
        image_bytes = BytesIO()
        image_wave_threshold.save(image_bytes, format='JPEG')
        image_bytes_wave_threshold = image_bytes.getvalue()
        image_wave_threshold.save('images/threshold_wave.png')

        wave_threshold_sko = sko(image, image_wave_threshold)

        image_array.append(image_bytes_wave_threshold)

        print("Threshold MATH:")
        image_math_threshold = utils.threshold(image_math, width, height)
        image_bytes = BytesIO()
        image_math_threshold.save(image_bytes, format='JPEG')
        image_bytes_math_threshold = image_bytes.getvalue()
        image_math_threshold.save('images/threshold_math.png')

        math_threshold_sko = sko(image, image_math_threshold)

        image_array.append(image_bytes_math_threshold)

        print("Sobel:")
        image_sobel = SobelOperator.sob_matrix(image, width, height)
        image_sobel_bytes = BytesIO()
        image_sobel.save(image_sobel_bytes, format='JPEG')
        image_sobel_bytes_bd = image_sobel_bytes.getvalue()
        image_sobel.save('images/sobel.png')

        sobel_sko = sko(image, image_sobel)

        image_array.append(image_sobel_bytes_bd)

        self.data.emit(
            {
                'date': date,
                'filename': filename,
                'image_array': image_array,
                'sko': [
                    dog_sko, wave_sko, math_sko, dog_threshold_sko, wave_threshold_sko, math_threshold_sko, sobel_sko
                ]
            }
        )

    def __del__(self):
        self.wait()


class DataBaseThread(QThread):
    def __init__(self):
        super().__init__()
        self.data: dict = {}
        self.result_labels: list = []
        self.table = None

    def setData(self, data, result_labels, table):
        self.data = data
        self.result_labels = result_labels
        self.table = table

    def run(self) -> None:
        date = self.data['date']
        filename = self.data['filename']
        images = self.data['image_array']
        sko = self.data['sko']

        # Создание подключения к базе данных
        with sqlite3.connect('db.sqlite3') as conn:
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
                                        sko REAL,
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
            # conn.commit()
            cursor.execute(
                "INSERT INTO image (name, data, created_at) VALUES (?, ?, ?)",
                (filename, images[0], date)
            )

            fk_id = cursor.lastrowid

            cursor.executemany(
                "INSERT INTO processed_image (image_id, name_method, sko, data, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (fk_id, "dog", sko[0], images[1], date),
                    (fk_id, "wave", sko[1], images[2], date),
                    (fk_id, "threshold_dog", sko[2], images[3], date),
                    (fk_id, "math", sko[3], images[4], date),
                    (fk_id, "threshold_wave", sko[4], images[5], date),
                    (fk_id, "threshold_math", sko[5], images[6], date),
                    (fk_id, "sobel", sko[6], images[7], date)
                ]
            )
            sleep(1)
            conn.commit()

            for i in range(7):
                self.table.setItem(i, 0, QTableWidgetItem(str(sko[i].astype('float'))))
            max_sko = max(sko)
            for i in range(7):
                self.table.setItem(i, 1, QTableWidgetItem(str(100 - ((sko[i] / max_sko) * 100).astype('float'))))

            cursor.execute('''
            SELECT processed_image.data
            FROM image
            JOIN processed_image ON image.id = processed_image.image_id
            WHERE image.id = (SELECT MAX(id) FROM image)
            ''')
            images_data = cursor.fetchall()
            for image_data, label in zip(
                    images_data, self.result_labels
            ):
                image = BytesIO(image_data[0])
                pixmap = QPixmap()
                pixmap.loadFromData(image.read())
                label.setPixmap(pixmap)

            conn.commit()


class OutputLogger(QObject):
    emit_write = pyqtSignal(str, int)

    class Severity:
        DEBUG = 0
        ERROR = 1

    def __init__(self, io_stream, severity):
        super().__init__()

        self.io_stream = io_stream
        self.severity = severity

    def write(self, text):
        self.io_stream.write(text)
        self.emit_write.emit(text, self.severity)

    def flush(self):
        self.io_stream.flush()


OUTPUT_LOGGER_STDOUT = OutputLogger(sys.stdout, OutputLogger.Severity.DEBUG)
OUTPUT_LOGGER_STDERR = OutputLogger(sys.stderr, OutputLogger.Severity.ERROR)

sys.stdout = OUTPUT_LOGGER_STDOUT
sys.stderr = OUTPUT_LOGGER_STDERR


class RuntimeStylesheets(QMainWindow, QtStyleTools):

    def __init__(self):
        super().__init__()
        self.main = uic.loadUi('ui/untitled.ui', self)
        self.add_menu_theme(self.main, self.main.menuStyles)


class App(QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам и т.д. в файле design.py
        super().__init__()

        # Это нужно для инициализации нашего дизайна
        self.setupUi(self)

        # Кнопка старт и функция реагирующая на нажатия и перенаправляющая в метод start_script
        self.pushButton.clicked.connect(self.start_script)

        # Консоль
        self.console_text = self.label_11

        # Кнопка выбора картинки и функция реагирующая на нажатия и перенаправляющая в метод start_script
        self.pushButton_2.clicked.connect(self.browse_folder)  # Выполнить функцию browse_folder

        # Инициализируем поток
        self.thread = None

        # Поля для картинок (результат)
        self.result_labels = (
            self.label_5, self.label_6, self.label_7, self.label_8, self.label_9, self.label_10, self.label_13
        )

        # Подписи в левом-верхнем углу картинок
        self.image_names = (
            "dog", "wave", "math", "dog_threshold", "wave_threshold", "math_threshold"
        )

        self.coords = ((580, 70), (780, 70), (580, 250), (780, 250), (580, 430), (780, 430))
        self.name_labels = [QLabel(self.image_names[_], self) for _ in range(6)]
        [name_label.move(*coord) for coord, name_label in zip(self.coords, self.name_labels)]
        for qlabel in self.name_labels:
            # Получаем рекомендуемый размер виджета на основе содержимого
            size = qlabel.sizeHint()

            # Устанавливаем фиксированную ширину на основе рекомендуемого размера
            qlabel.setFixedWidth(size.width())

        # Выбранная оригинальная картинка
        self.orig_photo = self.label

        # Секундомер в левом нижнем углу
        self.time = QTime(0, 0, 0)
        self.timer = QTimer()
        self.timer_canvas = self.label_14

        # Таблица
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget.setStyleSheet("QTableWidget {border: 1px solid yellow;}")

        # Кнопка Гаусса
        self.pushButton_4.clicked.connect(self.change_noize)

        # Картинка + ориг
        self.orig_image = None
        self.image_path = None
        self.image = None

        # Кнопка возврата к оригиналу
        self.pushButton_3.clicked.connect(self.return_to_orig)

        # SCALE
        self.scale_value = 0

        # API_MODEL
        try:
            rf = Roboflow(api_key="JpeOSShd0Un7h7ml3dzj")
            project = rf.workspace().project("car_-w0bxi")
            self.model = project.version(1).model
            print('Succesfully connected to model!')
        except Exception as e:
            print(e)
            self.model = None

    def return_to_orig(self):
        if self.image:
            self.scale_value = 0
            self.image = self.orig_image
            img = self.orig_image
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)

            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue())

            self.orig_photo.setPixmap(pixmap)
        else:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("Ошибка")
            msgBox.setInformativeText("Картинка не выбрана!")
            msgBox.setWindowTitle("Ошибка")
            msgBox.exec_()

    def change_noize(self):
        if self.image:
            self.image = self.orig_image
            image, width, height, pixels, matrix = ImageManager.get_image_info2(self.image)
            scale = self.spinBox.value()
            if scale <= 50:
                scale *= 2
            else:
                step = (scale - 50) // 5
                scale *= (2 + step)
            mass_noise = MassNoize.gaussian_noise(image, width, height, scale)
            noize_pixmap = QPixmap.fromImage(
                QImage(
                    mass_noise.tobytes(),
                    mass_noise.size[0],
                    mass_noise.size[1],
                    QImage.Format_RGB888
                )
            )

            self.scale_value = self.spinBox.value()
            self.image = mass_noise
            self.orig_photo.setPixmap(noize_pixmap)
            self.orig_photo.setScaledContents(True)
        else:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("Ошибка")
            msgBox.setInformativeText("Картинка не выбрана!")
            msgBox.setWindowTitle("Ошибка")
            msgBox.exec_()

    def start_script(self):
        try:
            if self.orig_photo.pixmap():
                if self.thread is None:
                    for label in self.result_labels:
                        if not label:
                            label.setPixmap(None)
                    self.time = QTime(0, 0, 0)
                    self.timer.timeout.connect(self.timerEvent)
                    self.timer.start(1000)
                    self.pushButton_4.setEnabled(False)
                    self.pushButton_3.setEnabled(False)
                    self.thread = ScriptThread()
                    self.dbCommit = DataBaseThread()
                    self.thread.set_value(self.spinBox.value(), self.image, self.image_path)
                    self.thread.start()
                    self.thread.data.connect(self.dataDbUpdate)
                    OUTPUT_LOGGER_STDOUT.emit_write.connect(self.append_log)
                    OUTPUT_LOGGER_STDERR.emit_write.connect(self.append_log)
                else:
                    msgBox = QMessageBox()
                    msgBox.setIcon(QMessageBox.Warning)
                    msgBox.setText("Прерывание скрипта.")
                    msgBox.setInformativeText("Пожалуйста, дождитесь пока скрипт не завершит свою работу.")
                    msgBox.setWindowTitle("Прерывание скрипта.")
                    msgBox.exec_()
            else:
                msgBox = QMessageBox()
                msgBox.setIcon(QMessageBox.Warning)
                msgBox.setText("Ошибка")
                msgBox.setInformativeText("Картинка не выбрана!")
                msgBox.setWindowTitle("Ошибка")
                msgBox.exec_()
        except Exception as e:
            print(e)

    def dataDbUpdate(self, data):
        self.thread = None
        self.timer.stop()
        self.dbCommit.setData(data, self.result_labels, self.tableWidget)
        self.dbCommit.start()
        self.pushButton_4.setEnabled(True)
        self.pushButton_3.setEnabled(True)

        methods = {
            "orig": '',
            "dog": '',
            "wave": '',
            "math": '',
            "dog_threshold": '',
            "wave_threshold": '',
            "math_threshold": ''
        }
        for img, method in zip(data['image_array'], methods.keys()):
            char_count = {}
            # преобразование байтового изображения в numpy array
            nparr = np.frombuffer(img, np.uint8)

            # декодирование numpy array в cv2.image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            height, width = img.shape[:2]
            down_low = 2
            center_x, center_y = int(width / down_low), int(height / down_low)
            new_width, new_height = int(width / down_low), int(height / down_low)

            left_x = center_x - int(new_width / down_low)
            top_y = center_y - int(new_height / down_low)
            right_x = center_x + int(new_width / down_low)
            bottom_y = center_y + int(new_height / down_low)

            cropped_image = img[top_y:bottom_y, left_x:right_x]

            repeat = 1
            prev_rec = None
            this_rec = None
            for i in range(1, 14):
                try:
                    text = pytesseract.image_to_string(
                        cropped_image,
                        lang='eng',
                        config=f'--psm {i} --oem 3 -c tessedit_char_whitelist=0123456789'
                    )
                    if len(text.split()) > 1:
                        for rec in text.split():
                            this_rec = rec
                            if this_rec == prev_rec:
                                repeat *= 1
                            else:
                                repeat = 1
                            if rec in char_count:
                                char_count[rec] += 1 * repeat
                            else:
                                char_count[rec] = 1
                            prev_rec = rec
                        print(i, text.split())
                    if len(text.split()) == 1:
                        print(i, text.split())
                        if text.split()[0] in char_count:
                            this_rec = text.split()[0]
                            if this_rec == prev_rec:
                                repeat *= 4.2
                            else:
                                repeat = 1
                            char_count[text.split()[0]] += 1 * repeat
                            prev_rec = text.split()[0]
                        else:
                            this_rec = text.split()[0]
                            char_count[text.split()[0]] = 1
                            prev_rec = text.split()[0]


                except Exception as e:
                    ''
            for key in char_count.keys():
                if key.isdigit() and len(key) == 1:
                    char_count[key] /= 10
            max = sum([elem for elem in char_count.values()])
            char_count = sorted(char_count.items(), key=lambda x: float(x[1]), reverse=True)
            count_lst = [f"{key}: " + "{:.2f}%".format((value / max) * 100) for key, value in char_count]
            char_count2str = '\n'.join(count_lst)
            print(char_count2str)
            methods[method] = char_count2str
        print(methods)


        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText('Результаты распознавания')
        msgBox.setInformativeText("\n".join([f'{key}:\n{item}' for key, item in methods.items()]))
        msgBox.setWindowTitle("Распознавание")
        msgBox.exec_()

    def timerEvent(self) -> None:
        self.time = self.time.addSecs(1)
        self.timer_canvas.setText(self.time.toString())

    def append_log(self, text, severity):
        text = repr(text)

        if severity == OutputLogger.Severity.ERROR:
            text = '{}'.format(text)[3:]

        font = QtGui.QFont()
        font.setPointSize(15)
        if len(text) > 6:
            self.console_text.setText(text)
            self.console_text.setFont(font)

    def browse_folder(self):
        fileName = FileDialog(
            str(os.path.abspath('assets'))
        )

        if fileName:
            self.image_path = fileName
            if self.model:
                img = ImgCrop(fileName, self.model)
            else:
                img = Image.open(self.image_path)
            self.image = img
            self.orig_image = img

            buffer = BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)

            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue())

            self.orig_photo.setPixmap(pixmap)
            self.orig_photo.setScaledContents(True)


def main():
    app = QApplication(sys.argv)  # Новый экземпляр QApplication
    apply_stylesheet(app, theme='dark_amber.xml')
    window = App()  # Создаём объект класса ExampleApp
    window.setMinimumSize(window.width(), window.height())
    window.setMaximumSize(window.width(), window.height())
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
