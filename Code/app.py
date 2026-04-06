import cv2
import time
from core.pipeline import TrafficPipeline


def main():
    pipeline = TrafficPipeline()

    # Use video file or webcam
    cap = cv2.VideoCapture("/Users/akashsaha/Downloads/Soft_Computing_Project/Data/sample.mp4")
    # cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open video source")
        return
    print("Video source opened successfully")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed. Switching to webcam...")
            cap.release()
            cap = cv2.VideoCapture(0)
            continue

        # Resize for better performance
        frame = cv2.resize(frame, (960, 540))

        try:
            processed = pipeline.process_frame(frame)
            if processed is not None:
                frame = processed
        except Exception as e:
            print(f"Processing error: {e}")
            # keep showing the last good/raw frame

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {int(fps)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Traffic AI - Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pipeline.finalize()


if __name__ == "__main__":
    main()