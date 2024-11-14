import json
from PIL import Image
import pytesseract
import cv2
import os

# Установка пути к tessdata
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/tessdata/'

# Замените 'eng' на нужный язык, например 'rus' для русского
language = 'rus'
image_path = 'img2.jpg'
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

# Применение OCR с выбранным языком
text = pytesseract.image_to_string(Image.open(filename), lang=language)
os.remove(filename)

# Обнаружение текстовых объектов на изображении
boxes_info = []
detection_boxes = pytesseract.image_to_boxes(Image.open(image_path))
h, w, _ = image.shape

for box in detection_boxes.splitlines():
    b = box.split()
    x, y, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(image, (x, h-y), (x2, h-y2), (0, 255, 0), 2)
    boxes_info.append({
        'character': b[0],
        'coordinates': {
            'x1': x,
            'y1': h - y,
            'x2': x2,
            'y2': h - y2
        }
    })

# Запись результатов в JSON файл
results = {
    'text': text,
    'boxes': boxes_info
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

