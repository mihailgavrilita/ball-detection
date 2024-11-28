from argparse import ArgumentParser
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("accuracy.py")
logging.basicConfig(level=logging.NOTSET)


def calculate_distance(point1, point2):
    """ Calculates the distance between 2 points on the screen.

        Args:
            point1(dict): A point on the screen.
            point2(dict): Another point on the screen.

        Returns:
            float: The distance between the points.
    """
    return np.sqrt((point1['ball_x'] - point2['ball_x'])**2 + (point1['ball_y'] - point2['ball_y'])**2)


if __name__ == "__main__":
    logger.info("Reading arguments..")
    arg_parser = ArgumentParser()
    arg_parser.add_argument("labels", type=str, help="Path to csv with ball positions")
    arg_parser.add_argument("true_labels", type=str, help="Path to csv against which to compare")
    args = arg_parser.parse_args()

    labels_file = args.labels
    true_labels_file = args.true_labels

    logger.info("Opening CSVs..")
    df = pd.read_csv(labels_file)
    df.set_index("frame_no", inplace=True)

    true_df = pd.read_csv(true_labels_file)
    true_df.set_index("frame_no", inplace=True)

    logger.info("Setting metrics..")
    hits = 0
    near = 0
    miss = 0
    false_positives = 0

    logger.info("Calculating statistics..")
    for index, row in df.iterrows():
        if index in true_df.index:
            distance = calculate_distance(row, true_df.loc[index])

            hits += 1 if distance <= 5 else 0
            near += 1 if distance > 5 and distance <= 10 else 0
            miss += 1 if distance > 10 else 0

        else:
            false_positives += 1

    logger.info(f"""Calculation finished. Results:
        Hits:\t {hits} ({hits / len(df):.2f});
        Near:\t {near} ({near / len(df):.2f});
        Miss:\t {miss} ({miss / len(df):.2f});
        F pos:\t {false_positives} ({false_positives / len(df):.2f}).
    """)
