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

    logger.info("Preparing classifier..")
    ball_cascade = cv2.CascadeClassifier("classifiers/cascade.xml")

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

        # New
        gray = cv2.medianBlur(gray, 9)
        rows = gray.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 16,
                                   param1=300, param2=20, minRadius=0, maxRadius=10)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for idx, i in enumerate(circles[0, :]):
                center = (i[0], i[1])
                # circle center
                # cv2.circle(gray, center, 1, (255, 0, 0), 7)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(gray, f"{idx}", center, font, 0.5, (11, 255, 255), 3, cv2.LINE_AA)
                # circle outline
                # radius = i[2]
                # cv2.circle(frame, center, radius, (255, 0, 255), 3)

                df.loc[frame_no] = [frame_no, center[0], center[1]]
                break

        # balls = ball_cascade.detectMultiScale(gray, 10, 10, maxSize=(10, 10))

        # for (x, y, w, h) in balls:
        #     ball_x = x + w // 2
        #     ball_y = y + h // 2

        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
        #     df.loc[frame_no] = [frame_no, ball_x, ball_y]
        #     # break

        if is_visualize:
            show_img = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("Frame", show_img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    df = df.astype(int)
    df.to_csv(labels_file, index=False)
    logger.info(f"Predicted ball coordinates saved to {labels_file}.")
