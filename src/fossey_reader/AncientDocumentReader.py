
import cv2
import os


class AncientDocumentReader:

    def __init__ (self, document_directory):
        self.doc_dir = document_directory

    
    def load_images_from_folder(self):
        images = []
        for filename in os.listdir(self.doc_dir):
            print(filename)
            img = cv2.imread(os.path.join(self.doc_dir, filename))
            if img is not None:
                images.append(img)
        return images