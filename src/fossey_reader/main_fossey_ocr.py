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

    # This dataset actually has 252 images, due to copying over accident
    folder_100 = "dataset_100/"

    #my_images = load_images_from_folder(folder)
    my_images_100 = load_images_from_folder(folder_100)


    # fix 1089a for that 5 connected to the table line
    # 1089a
    #my_test_im = my_images[0]
    #my_test_im = my_images[3]
    #my_test_im = my_images[2]

    # this one also has index detection as part of the lines problem
    #my_test_im = my_images[5]

    # image[6] is a big image with a broken bouding box
    #my_test_im = my_images[7]
    #fn = my_test_im['filename']
    #fn = fn[:-4]
    #print(fn)

    #myAncientReader = AncientDocumentReader(my_test_im)
    ##myAncientReader.clean_image()
    #no_line_folder = "tab_lines_removed/"
    no_line_folder_100 = "tab_lines_removed_100/"
    no_line_folder_100_bound = "tab_lines_to_remove_100/"
    #myAncientReader.removeBoundingLines()

    #cv2.imwrite(no_line_folder_100+fn+"_bounding_line_to_remove.jpg", myAncientReader.boundingLines)
    #cv2.imwrite(no_line_folder_100+fn+"_bounding_line_removed.jpg", myAncientReader.lineCleaned)

    # Now let's run it on the slightly bigger sample size
    for i,im in enumerate(my_images_100):
        print("Processing number " + str(i+1) + " of 252")
        fn = im['filename']
        fn = fn[:-4]
        #print(fn)

        my_ancient_reader = AncientDocumentReader(im)
        my_ancient_reader.removeBoundingLines()

        cv2.imwrite(no_line_folder_100_bound+fn+"_bounding_line_to_remove.jpg", my_ancient_reader.boundingLines)
        cv2.imwrite(no_line_folder_100+fn+"_bounding_line_removed.jpg", my_ancient_reader.lineCleaned)
