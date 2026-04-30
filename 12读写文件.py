import cv2
import datetime
import time
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

TARGETS = ['person', 'cup', 'cell phone']
LINE_Y = 300
last_side = {}
alert_timer = 0
last_log_time = 0
alert_cooldown = {}  # 报警冷却

log_file = open('detection_log.txt', 'w', encoding='utf-8')
log_file.write("=== Detection Start ===\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    for box in results[0].boxes:
        cls_id = int(box.cls)
        label = model.names[cls_id]

        if label not in TARGETS:
            continue

        conf = float(box.conf)
        if conf < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # 普通记录，每秒一次
        now = time.time()
        if now - last_log_time > 1:
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')
            log_file.write(f"{timestamp} | {label} | conf:{conf:.2f} | pos:({cx},{cy})\n")
            last_log_time = now

        # 判断绊线
        side = 'above' if cy < LINE_Y else 'below'
        key = f"{cx // 50}"

        if key in last_side and last_side[key] != side:
            now_alert = time.time()
            if key not in alert_cooldown or now_alert - alert_cooldown[key] > 5:
                alert_timer = 30
                timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                log_file.write(f"{timestamp} | !! ALERT !! | {label} crossed line at ({cx},{cy})\n")
                log_file.flush()
                alert_cooldown[key] = now_alert

        last_side[key] = side

    # 画绊线
    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 255, 255), 2)

    # 报警
    if alert_timer > 0:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]),
                      (0, 0, 255), 8)
        cv2.putText(frame, '!! ALERT !!', (150, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        alert_timer -= 1

    cv2.imshow('YOLO', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()