from argparse import ArgumentParser
import cv2
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("solution.py")
logging.basicConfig(level=logging.NOTSET)

if __name__ == "__main__":
    logger.info("Reading arguments..")
    arg_parser = ArgumentParser()
    arg_parser.add_argument("video", type=str, help="Path to video file")
    arg_parser.add_argument("--visualize", help="Visualize frame by frame detections", action="store_true")
    args = arg_parser.parse_args()

    video_file = args.video
    labels_file = "".join(video_file.split(".")[:-1]) + ".csv"
    is_visualize = args.visualize

    logger.info("Loading video..")
    cap = cv2.VideoCapture(video_file)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info("Initializing dataframe..")
    columns = ["frame_no", "ball_x", "ball_y"]
    df = pd.DataFrame(columns=columns)

    logger.info("Predicting..")
    for frame_no in range(num_frames):
        if frame_no % 100 == 0:
            logger.info(f"Frame {frame_no}\tout of {num_frames}.")

        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 9)

        rows = gray.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 16,
                                   param1=300, param2=20, minRadius=0, maxRadius=10)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            circle = circles[0, :][0]

            ball_x = circle[0]
            ball_y = circle[1]

            cv2.circle(frame, (ball_x, ball_y), 1, (255, 0, 255), 11)
            df.loc[frame_no] = [frame_no, ball_x, ball_y]

        if is_visualize:
            show_img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("Frame", show_img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    df = df.astype(int)
    df.to_csv(labels_file, index=False)
    logger.info(f"Predicted ball coordinates saved to {labels_file}.")
