import cv2
from ultralytics import YOLO
import time

def main():
    # YOLOv8モデル（初回は自動DL）
    model = YOLO("yolov8n.pt")  # 軽量モデル（まずはこれ推奨）

    # Webカメラ開始
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webカメラを開けませんでした")
        return

    # FPS計算用
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 推論（frameをそのまま渡せる）
        results = model.predict(frame, imgsz=640, conf=0.4, verbose=False)

        # 描画済みフレームを取得
        annotated_frame = results[0].plot()

        # FPS表示
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 表示
        cv2.imshow("YOLOv8 Webcam", annotated_frame)

        # qで終了
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
