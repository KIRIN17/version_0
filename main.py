import cv2
#для детектирования объектов на видео,фон которого статичен

cap = cv2.VideoCapture("highway_traffic.mp4")
obj_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    _, img = cap.read()
    roi = img[1:int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 1:int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))]

    mask = obj_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)  # 254 - так как хотим очистить маску от серого

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:#размер площади(в пикселях),начиная с которого мы начинаем отслеживать объект
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Mask", mask)
    cv2.imshow("Img", img)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
