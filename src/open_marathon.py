import matplotlib.pyplot as plt
from face_detection import cropped_images
from ocr import ada_ocr


def main():
    # Matplotlib Configurations
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    cropped = cropped_images(image_path='../data/015.jpg', see_full=False)
    predicted = ada_ocr(images=cropped)
    print(predicted)
    # for crop in cropped:
    #     plt.imshow(crop, cmap='gray')
    #     plt.show()


if __name__ == '__main__':
    main()
