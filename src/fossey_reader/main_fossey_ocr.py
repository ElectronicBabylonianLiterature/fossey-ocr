
import json
import cv2
import os

import AncientDocumentReader


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        print(filename)
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


if __name__ == '__main__':
    # Let's import some images, test sample first
    print("We're doing stuff here now")


    # TODO: pass the dataset folder as an argument or something?
    folder = "dataset/"

    myImages = load_images_from_folder(folder)