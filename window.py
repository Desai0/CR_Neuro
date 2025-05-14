import numpy as np
import cv2
import mss
import time

monitor_details = {"top": 35, "left": 2674, "width": 766, "height": 1355}

title = "Clash Royale Screen Capture"
sct = mss.mss()

print("Нажмите 'q' в окне с захватом, чтобы выйти.")

while True:
    # Захват кадра
    sct_img = sct.grab(monitor_details)

    # Преобразование в формат OpenCV (BGR)
    # mss возвращает BGRA
    img = np.array(sct_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Отображение кадра
    cv2.imshow(title, img)
    # fps
    time.sleep(0.033) # ~30 FPS

    # Выход 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

print("Скрипт завершен.")