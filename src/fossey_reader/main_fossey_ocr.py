import cv2
import os
from AncientDocumentReader import AncientDocumentReader


def load_images_from_folder(doc_dir):
    images = []
    for filename in os.listdir(doc_dir):
        #print(filename)
        img = cv2.imread(os.path.join(doc_dir, filename))
        if img is not None:
            images.append({"image": img, "filename": filename})
    return images


if __name__ == '__main__':

    # TODO: pass the dataset folder as an argument or something?
    folder = "dataset/"

    myImages = load_images_from_folder(folder)

    myTestIm = myImages[3]
    #print(myTestIm['filename'])

    myAncientReader = AncientDocumentReader(myTestIm)
    myAncientReader.clean_image()