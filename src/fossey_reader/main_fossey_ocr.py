
from AncientDocumentReader import AncientDocumentReader




if __name__ == '__main__':

    # TODO: pass the dataset folder as an argument or something?
    folder = "dataset/"
    myAncientReader = AncientDocumentReader(folder)

    myImages = myAncientReader.load_images_from_folder()