import json
from PIL import Image
import pytesseract
import cv2
import os

# Установка пути к tessdata
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/tessdata/'

# Замените 'eng' на нужный язык, например 'rus' для русского
language = 'rus'
image_path = 'img.jpg'
preprocess = "thresh"

# Загрузить изображение и преобразовать его в оттенки серого
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применить пороговую обработку или медианное размытие
if preprocess == "thresh":
    gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
elif preprocess == "blur":
    gray = cv2.medianBlur(gray, 3)

# Сохранить временный файл в оттенках серого
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

# Применение OCR с выбранным языком и получение данных о распознанных словах
data = pytesseract.image_to_data(Image.open(filename), lang=language, output_type=pytesseract.Output.DICT)
os.remove(filename)

# Объединение символов в слова
words = []
current_word = ""
for i in range(len(data['text'])):
    try:
        conf_value = float(data['conf'][i])  # Преобразуем значение уверенности в float
    except ValueError:
        conf_value = -1  # Если преобразование не удалось, устанавливаем значение уверенности в -1

    if conf_value > 0:  # Проверка на уверенность распознавания
        if data['text'][i].strip():  # Если текст не пустой
            current_word += data['text'][i] + " "
        else:
            if current_word:  # Если текущая строка не пустая, добавляем в список слов
                words.append(current_word.strip())
                current_word = ""
if current_word:  # Добавляем последнее слово, если оно есть
    words.append(current_word.strip())

# Запись результатов в JSON файл с нужным форматом (каждое слово на новой строке)
results = {
    'words': words  # Список слов будет записан как массив
}

with open('results.json', 'w', encoding='utf-8') as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)

# Показ выходных изображений
cv2.imshow("Image", image)
cv2.imshow("Output", gray)

# Закрытие окон только по нажатию 'q'
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()