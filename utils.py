from mnist import MNIST
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import PIL.ImageOps

def load_mnist():
    mndata = MNIST('samples')

    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    train_images, train_labels, test_images, test_labels = np.array(
        train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)

    train_images = np.reshape(train_images/255, (len(train_images), 28, 28))
    test_images = np.reshape(test_images/255, (len(test_images), 28, 28))
    return (train_images, train_labels), (test_images, test_labels)


def load_custom():
    images = []
    for i in range(10):
        image = PIL.ImageOps.invert(Image.open(f"custom-images/custom{i}.png").convert('L'))
        images.append(np.array(image))
    return np.array(images)


def view_img(img, label="", xlabel=""):
    plt.grid(False)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xlabel(xlabel)
    plt.title(label)
    plt.show()


