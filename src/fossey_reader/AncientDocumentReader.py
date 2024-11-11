
import cv2
import numpy as np


# Perform image processing operations on a single image
class AncientDocumentReader:

    def __init__ (self, im_to_process):
        self.image = im_to_process['image']
        self.filename = im_to_process['filename']


    def showImage(self, img):
        # Utility for showing image in various stages of processing
        # Using filename for manual comparison with original image
        window_name = self.filename[:-4] + " in Progress"
        cv2.imshow(window_name, img)
        cv2.waitKey()


    def cleanImage (self):
        # Remove the horizontal and vertical lines in Fossey that indicate tabular divisions
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # ret is the threshold value calculated by Otsu's method
        # Using global threshold since all images were scanned under similar lighting conditions
        # Adaptive threshold results in a lot of noise
        ret, thresh_im = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.threshIm = thresh_im
        #thresh_im = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        #self.showImage(gray)

        # Currently selecting an elliptical structural element of 4,4 because it has the best noise reduction 
        # with little seeming impact on the bibliographical text (or cuneiform signs) that will
        # later be OCR'd so may need to come back to reevaluate considering that the cuneiform will not need to be OCR processed
        # so the impact of the image pre-processing here is probably not to be taken too much into consideration

        # Look at https://docs.opencv.org/4.x/dd/dd7/tutorial_morph_lines_detection.html for the kernel as horizonal structure
        # Except instead of black lines for horizontal structure, use whitespace as the "horizontal structure"
        # can come back to this kernel and opening for noise reduction later

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))

        # Opening is erosion followed by dilation which is useful for removing noise
        opening = cv2.morphologyEx(thresh_im, cv2.MORPH_OPEN, kernel, iterations=1)
        self.opening = opening

        closing = cv2.morphologyEx(thresh_im, cv2.MORPH_CLOSE, kernel, iterations=1)
        self.closing = closing


    def removeBoundingLines(self):
        ######## Trying out a new approach where I jump into line extraction and/or separation without all the cleaning for now ###
        # Create the images that we will use to extract the horizontal and vertical lines
        self.cleanImage()

        # using closing because found an image with a white scanner line
        contours, hierarchies = cv2.findContours(self.closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print(hierarchies)


        # Can identify the bounding box we want to remove because it is the only contour with 
        # children

        #contours = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contours = contours[0] if len(contours) == 2 else contours[1]

        thing = np.copy(self.image)
        another = np.copy(self.closing)

        for c in contours:
            area = cv2.contourArea(c)
            #print(area)
            perimeter = cv2.arcLength(c,False)
            # print(perimeter)
            if perimeter > 8_000:
                cv2.drawContours(thing, [c], -1, (0,255,0), 10)

                if area > 1_000_000:
                    cv2.drawContours(another, [c], -1, (0,0,0), 10)
                else:
                    cv2.fillPoly(another, [c], color = (0,0,0))

        #self.showImage(thing)
        #self.showImage(another)
        self.boundingLines = thing
        self.lineCleanedInverted = another
        self.lineCleaned = cv2.bitwise_not(another)
        #self.showImage(self.lineCleaned)
