from __future__ import division
import cv2
import keras


def resize_image(img, min_side=600, max_side=1024):
    (rows, cols, _) = img.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, wich can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale


def preprocess_image(x):
    # mostly identical to
    # "https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already
    x = x.astype(keras.backend.floatx())
    if keras.backend.image_data_format() == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68
        else:
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
    else:
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x


