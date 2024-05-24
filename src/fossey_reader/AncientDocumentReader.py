
import cv2


# Perform image processing operations on a single image
class AncientDocumentReader:

    def __init__ (self, im_to_process):
        self.image = im_to_process['image']
        self.filename = im_to_process['filename']


    def show_image(self, img):
        # Utility for showing image in various stages of processing
        # Using filename for manual comparison with original image
        window_name = self.filename[:-4] + " in Progress"
        cv2.imshow(window_name, img)
        cv2.waitKey()


    def clean_image(self):
        # Remove the horizontal and vertical lines in Fossey that indicate tabular divisions
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # ret is the threshold value calculated by Otsu's method
        # Using global threshold since all images were scanned under similar lighting conditions
        # Adaptive threshold results in a lot of noise
        ret, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #thresh_im = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Currently selecting an elliptical structural element of 4,4 because it has the least impact on cuneiform degradation
        # A size of 6,6 has the best noise reduction with little seeming impact on the bibliographical text that will
        # later be OCR'd so may need to come back to reevaluate considerinf that the cuneiform will not need to be OCR processed
        # so the impact of the image pre-processing here is probably not to be taken too much into consideration
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))

        # Opening is erosion followed by dilation which is useful for removing noise
        opening = cv2.morphologyEx(thresh_im, cv2.MORPH_OPEN, kernel, iterations=1)
        self.show_image(opening)
    