import sys
import matplotlib.pyplot as plt
from src.face_detection import cropped_images
from src.ocr import ada_ocr, ada_ocr_v2

def image_collection(file_path=None):
    if file_path == None:
        file_path = '../data/test_file.txt'
    file = open(file_path, 'r')
    file_list = file.readlines()
    print(file_list)

def main():
    # Initialization
    processed_data = {}
    # Matplotlib Configurations
    # fig = plt.figure()
    # ax = fig.add_subplot(111, aspect='equal')

    # Image file configuration
    image_path = sys.argv[1]
    filename = image_path.split('/')[-1]

    cropped = cropped_images(image_path=image_path, see_full=False)
    predicted = ada_ocr_v2(images=cropped, debug=True)

    # Save file
    processed_data[filename] = predicted
    print(processed_data)

    for crop in cropped:
        plt.imshow(crop, cmap='gray')
        plt.show()


if __name__ == '__main__':
    main()
