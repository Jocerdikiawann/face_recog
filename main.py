import cv2
import time
from threading import Thread
from face_recog import FaceRecognitionSystem

def run_video_recognition(face_system):
    """Run optimized video recognition."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    face_system.is_processing = True
    process_thread = Thread(target=face_system.process_frame_async)
    process_thread.start()

    last_process_time = time.perf_counter()
    processing_interval = 0.1  # Adjust processing interval (in seconds)
    min_confidence_threshold = 50  # Ignore matches with confidence below this value

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the camera.")
            break

        current_time = time.perf_counter()

        # Retrieve results from the async result queue
        if not face_system.result_queue.empty():
            face_system.last_results = face_system.result_queue.get()

        # Process frame if interval has passed
        if current_time - last_process_time > processing_interval:
            if face_system.processing_queue.empty():
                face_system.processing_queue.put(frame.copy())
                last_process_time = current_time

        # Draw results on the frame
        for result in face_system.last_results:
            x, y, w, h = result['bbox']
            person = result['person']
            confidence = result['confidence']

            # Only display matches above confidence threshold
            if confidence >= min_confidence_threshold:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if person:
                    text = f"{person} ({confidence:.1f}%)"
                    cv2.putText(frame, text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Face Recognition', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    face_system.is_processing = False
    process_thread.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    dataset_path = "./face_dataset"  # Specify your dataset path

    print("Initializing Face Recognition System...")
    face_system = FaceRecognitionSystem(dataset_path)
    face_system.build_database()

    print("Starting Video Recognition...")
    run_video_recognition(face_system)
