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

    my_images = load_images_from_folder(folder)


    # fix 1089a for that 5 connected to the table line
    # 1089a
    my_test_im = my_images[0]
    #my_test_im = my_images[3]
    #my_test_im = my_images[2]

    # this one also has index detection as part of the lines problem
    #my_test_im = my_images[5]


    # image[6] is a big image with a broken bouding box
    #my_test_im = my_images[7]
    fn = my_test_im['filename']
    fn = fn[:-4]
    print(fn)

    myAncientReader = AncientDocumentReader(my_test_im)
    #myAncientReader.clean_image()
    no_line_folder = "tab_lines_removed/"
    myAncientReader.removeBoundingLines()


    cv2.imwrite(no_line_folder+fn+"_bounding_line_to_remove.jpg", myAncientReader.boundingLines)
    cv2.imwrite(no_line_folder+fn+"_bounding_line_removed.jpg", myAncientReader.lineCleaned)
