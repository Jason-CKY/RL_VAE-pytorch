import numpy as np

def get_img(output):
    img = ((output * 0.5) + 0.5) * 255
    img = img.numpy().astype(np.uint8)
    if len(img.shape) == 3:
        img = img.transpose([1, 2, 0])
    elif len(img.shape) == 4:
        img = img.transpose([0, 2, 3, 1])
    return img
