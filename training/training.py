import os
import numpy as np
import pandas as pd
import cv2
import logging

logger = logging.getLogger("training.py")
logging.basicConfig(level=logging.NOTSET)

MIN_CROPS = 2
MAX_CROPS = 5
CROP_SIZE = (100, 100)


def random_coords(frame_width, frame_height):
    """ Generates a number of random coordinates, given the dimensions of a frame.

        The number of coordinates generated should be between MIN_CROPS and MAX_CROPS.
        The coordinates should allow a crop of CROP_SIZE to be sliced afterwards.

        Args:
            frame_width(int): Width of the frame.
            frame_height(int): Height off the frame.

        Returns:
            generator(tuple): The coordinates generated.
    """
    for _ in range(np.random.randint(MIN_CROPS, MAX_CROPS + 1)):
        yield (
            np.random.randint(0, frame_width - CROP_SIZE[1]),
            np.random.randint(0, frame_height - CROP_SIZE[0])
        )


def generate_negatives(cap, df_frames):
    logger.info("Selecting no ball frames..")
    no_ball_frames = [frame_no for frame_no in range(num_frames) if frame_no not in df_frames.index]
    np.random.shuffle(no_ball_frames)

    logger.info("Generating crops..")
    img_count = 1
    for frame_no in no_ball_frames:
        if img_count % 100 == 0:
            logger.info(f"Frame {img_count}\tout of {len(no_ball_frames)}.")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()

        if not ret:
            logger.info("Can't receive frame (stream end?). Exiting..")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        resized_image = cv2.resize(gray, (100, 100))
        img_count += 1
        cv2.imwrite(f"neg/{img_count}.jpg", resized_image)

        # for x, y in random_coords(frame_width, frame_height):
        #     # logger.info(f"x={x}\ty={y}")
        #     cv2.imwrite(f"neg/{img_count}.jpg", gray[y:y + CROP_SIZE[1], x:x + CROP_SIZE[0]])
        #     img_count += 1

        # if img_count >= 2000:
        #     break

    logger.info(f"Done. {img_count} negatives generated.")


def generate_negative_descriptor():
    for image_path in os.listdir("neg"):
        with open("bg.txt", "a") as f:
            f.write(f"neg/{image_path}\n")


def generate_positives(cap, df_frames):
    logger.info("Selecting ball frames..")
    ball_frames = list(df_frames.index)
    np.random.shuffle(ball_frames)

    logger.info("Generating crops..")
    img_count = 1
    for frame_no in ball_frames:
        if img_count % 100 == 0:
            logger.info(f"Frame {img_count}\tout of {len(ball_frames)}.")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()

        if not ret:
            logger.info("Can't receive frame (stream end?). Exiting..")
            break

        ball_x, ball_y = df_frames.loc[frame_no]
        logger.info(f"frame_no={frame_no}\tx={ball_x}\ty={ball_y}")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            cv2.imwrite(f"pos/{img_count}.jpg", gray[
                ball_y - CROP_SIZE[1] // 2:ball_y + CROP_SIZE[1] // 2,
                ball_x - CROP_SIZE[0] // 2:ball_x + CROP_SIZE[0] // 2
            ])
            img_count += 1
        except Exception as e:
            logger.error(e)

        if img_count >= 10000:
            break

    logger.info(f"Done. {img_count} positives generated.")


def generate_positive_descriptor():
    x, y, w, h = 45, 45, 10, 10
    for image_path in os.listdir("info"):
        with open("info/info.lst", "a") as f:
            f.write(f"{image_path} 1 {y} {x} {w} {h}\n")


if __name__ == "__main__":
    video_file = "../task/part1.mp4"
    labels_file = "../task/part1.csv"

    logger.info("Setting label index..")
    df_frames = pd.read_csv(labels_file)
    df_frames.set_index("frame_no", inplace=True)

    logger.info("Loading video..")
    cap = cv2.VideoCapture(video_file)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # logger.info("Generating negatives..")
    # generate_negatives(cap, df_frames)

    # logger.info("Generating negative descriptor..")
    # generate_negative_descriptor()

    # logger.info("Generating positives..")
    # generate_positives(cap, df_frames)

    # logger.info("Generating positive descriptor..")
    # generate_positive_descriptor()

    logger.info("Exiting.")
