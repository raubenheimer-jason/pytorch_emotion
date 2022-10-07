import torch
import numpy as np
from tqdm import tqdm
import os
import cv2


class EmotionData():
    # check that images are: 48 x 48 and grayscale
    # convert folder categories to 1-hot encoding

    TEST = "dataset\\test"
    TRAIN = "dataset\\train"
    DATA = [TEST, TRAIN]

    training_data = []
    test_data = []

    # np.eye(2)[self.LABELS[label]]

    EMOTIONS = {"angry": None, "disgust": None, "fear": None,
                "happy": None, "neutral": None, "sad": None, "surprise": None}

    # # make a copy of emotions dict
    # EMOTION_COUNTS = {TEST: {key: 0 for key, value in EMOTIONS.items()},
    #                   TRAIN: {key: 0 for key, value in EMOTIONS.items()}}

    # make a copy of emotions dict
    EMOTION_COUNTS = {TEST: {key: 0 for key, _ in EMOTIONS.items()},
                      TRAIN: {key: 0 for key, _ in EMOTIONS.items()}}

    for i, emotion in enumerate(EMOTIONS):
        # print(emotion)
        EMOTIONS[emotion] = np.eye(len(EMOTIONS))[i]

    def print_encoding(self):
        for emotion in self.EMOTIONS:
            # print(f"{emotion}\t{np.argmax(self.EMOTIONS[emotion])}")
            print(f"{emotion}\t{self.EMOTIONS[emotion]}")

    def make_data(self):
        """ training and test data """

        for data in self.DATA:
            for emotion in self.EMOTIONS:
                path = os.path.join(data, emotion)
                # for f in tqdm(os.listdir(path)):
                for f in (pbar := tqdm(os.listdir(path))):
                    pbar.set_description(path)
                    try:
                        img_path = os.path.join(path, f)
                        # print(img_path)
                        img = cv2.imread(img_path)
                        # cv2.imshow("emotion", img)
                        # cv2.waitKey(0)
                        # break
                        self.EMOTION_COUNTS[data][emotion] += 1

                    except Exception as e:
                        print(str(e))
                # break

    def print_balance(self):
        for e in self.EMOTION_COUNTS:
            print(f"{e}\t{self.EMOTION_COUNTS[e]}")


ed = EmotionData()
# ed.print_encoding()
ed.print_balance()
ed.make_data()
ed.print_balance()
