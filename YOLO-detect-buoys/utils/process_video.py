"""The function to use YOLOv8 to process a video."""

from collections import defaultdict

from cv2 import VideoCapture, destroyAllWindows, imshow, polylines, waitKey
from numpy import hstack, int32
from pafy import new
from ultralytics import YOLO


def process_video(url: str, model: YOLO) -> None:
    """Process a video with YOLOv8 tracking.

    Args:
        url (str): The URL of the video to process.
        model (YOLO): The YOLOv8 model to use for tracking.

    Returns:
        None: This function saves the processed video locally.

    Example:
        process_video("https://youtu.be/4WGpIOwkLA4?feature=shared", model)
    """
    video = new(url)
    best = video.getbest(preftype="mp4")

    cap = VideoCapture(best.url)

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            # https://docs.ultralytics.com/modes/track/
            results = model.track(frame, persist=True)

            # makes sure the boxes and track IDs are not None
            if results[0].boxes is not None and results[0].boxes.id is not None:
                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x_val, y_val, _, _ = box
                    track = track_history[track_id]
                    track.append((float(x_val), float(y_val)))  # x, y center point
                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = (
                        hstack(track).astype(int32).reshape((-1, 1, 2))
                    )  # finally this works! 😵‍💫

                    polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=(230, 230, 230),
                        thickness=10,
                    )
            else:
                annotated_frame = frame

            # Display the annotated frame
            imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    destroyAllWindows()
