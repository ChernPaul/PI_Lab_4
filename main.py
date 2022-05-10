# This is a sample Python script.

import numpy as np
from skimage.io import imsave, imshow, show, imread
from matplotlib import pyplot as plt
import random
from scipy import signal
from skimage.util import random_noise
from math import floor, sqrt


MAX_BRIGHTNESS_VALUE = 255
MIN_BRIGHTNESS_VALUE = 0
IMAGE_HEIGHT = 64
IMAGE_LENGTH = 64

WHITE_NOISE_MORE_PARAMETER = 0.1
WHITE_NOISE_PARAMETER = 0.25

MODEL1 = np.array([[1, 1, 1],
                   [0, 1, 0],
                   [0, 1, 0]])

MODEL2 = np.array([[0, 1, 0],
                   [0, 1, 0],
                   [1, 1, 1]])


def border_processing_function(element_value, border_val):
    if element_value >= border_val:
        return MAX_BRIGHTNESS_VALUE
    else:
        return MIN_BRIGHTNESS_VALUE


def border_processing(img_as_arrays, border_val):
    vector_img = np.vectorize(border_processing_function)
    new_img = vector_img(img_as_arrays, border_val)
    return new_img


def correct_limits_function(element_value):
    if element_value < MIN_BRIGHTNESS_VALUE:
        return MIN_BRIGHTNESS_VALUE
    if element_value > MAX_BRIGHTNESS_VALUE:
        return MAX_BRIGHTNESS_VALUE
    return element_value


def check_and_correct_limits(img_as_arrays):
    vector_img = np.vectorize(correct_limits_function)
    new_img = vector_img(img_as_arrays)
    return new_img


def create_random_objects(height_img, width_img, object_model, number_of_objects):
    img = np.zeros((height_img, width_img))
    for k in range(0, number_of_objects):
        i = random.randint(0, 63)
        j = random.randint(0, 63)
        img[i][j] = 64
    img = signal.convolve2d(img, object_model, boundary="symm", mode="same")
    img = img + 96
    return img


def linear_contrast_function(img, fmin, fmax):
    if img < fmin:
        img = 0
    elif img > fmax:
        img = 255
    else:
        img = (255*img - 255*fmin)/(fmax-fmin)
    return img


def lineary_contrast(img, fmax):
    fmin = img.min()
    lin = np.vectorize(linear_contrast_function)
    contrast_img = lin(img, fmin, fmax)
    return contrast_img


def Correlator(img, model):
    t = np.array(model) * 1/np.sum(np.square(model))
    new_img = list(img.copy().astype(float))
    mean_val = np.mean(new_img)
    size = np.shape(new_img)
    help_array = np.array(np.ones(len(new_img[0]))*mean_val)
    new_img.insert(0, help_array)
    new_img.append(help_array)
    for i in range(0, size[0] + 2, 1):
        new_img[i] = list(new_img[i])
        new_img[i].insert(0, mean_val)
        new_img[i].append(mean_val)
    new_img = np.array(new_img)
    reserve_img = img.copy().astype(float)

    for i in range(1, size[0] + 1, 1):
        for j in range(1, size[1] + 1, 1):
            x = new_img[i-1:i+2, j-1:j+2]  # mask 3x3
            sum_of_sqr = 0
            for row in x:
                for el in row:
                    sum_of_sqr += el*el
            if sum_of_sqr == 0:
                sum_of_sqr = 1
            b_correlation_function = 0
            for k in range(0, len(x), 1):
                for l in range(0, len(x[0]), 1):
                    b_correlation_function += x[k][l] * t[k][l]
            b_correlation_function /= sqrt(sum_of_sqr)
            reserve_img[i-1][j-1] = int(b_correlation_function*255)
    return reserve_img


def show_source_imgs(noise1, noise2, img_with_T, img_with_TandL):
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(2, 2, 1)
    plt.title("Noise image for T-object")
    imshow(noise1, cmap='gray')  # , vmin=0, vmax=255
    fig.add_subplot(2, 2, 2)
    plt.title("Noise image for T & Inverse T-object")
    imshow(noise2, cmap='gray')  # , vmin=0, vmax=255
    fig.add_subplot(2, 2, 3)
    plt.title("T-object image")
    imshow(img_with_T, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 2, 4)
    plt.title("T & Inverse T-object image")
    imshow(img_with_TandL, cmap='gray', vmin=0, vmax=255)
    return fig


def show_fields(img_with_objects, correlate_field_objects, bordered_img_of_field_objects, noise_img_with_objects,
                correlate_field_noise_objects, bordered_img_of_field_noise_objects, model_name):
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(2, 3, 1)
    plt.title("objects without noise")
    imshow(img_with_objects, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 2)
    plt.title(f"Correlated field of objects with {model_name}")
    imshow(correlate_field_objects, cmap='gray') # , vmin=0, vmax=255
    fig.add_subplot(2, 3, 3)
    plt.title("Bordered processing correlated field of objects")
    imshow(bordered_img_of_field_objects, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 4)
    plt.title("objects with noise")
    imshow(noise_img_with_objects, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 5)
    plt.title(f"Correlated field of objects with {model_name}")
    imshow(correlate_field_noise_objects, cmap='gray') # , vmin=0, vmax=255
    fig.add_subplot(2, 3, 6)
    plt.title("Bordered processing correlated field of objects")
    imshow(bordered_img_of_field_noise_objects, cmap='gray', vmin=0, vmax=255)
    return fig


if __name__ == '__main__':

    n = 10  # количество объектов
    img_with_T_random_positions = create_random_objects(IMAGE_HEIGHT, IMAGE_LENGTH, MODEL1, n)
    dispersion_of_img_with_T = np.var(img_with_T_random_positions)

    img_with_T_random1 = create_random_objects(IMAGE_HEIGHT, IMAGE_LENGTH, MODEL1, n)
    img_with_L_random1 = create_random_objects(IMAGE_HEIGHT, IMAGE_LENGTH, MODEL2, n)
    img_with_T_and_L_random = (img_with_T_random1 + img_with_L_random1)/2
    dispersion_of_img_with_T_and_L = np.var(img_with_T_and_L_random)
    avg_dispersion = (dispersion_of_img_with_T + dispersion_of_img_with_T_and_L) / 2

    white_noise = np.random.normal(loc=0, scale=float(sqrt(avg_dispersion/ WHITE_NOISE_PARAMETER)),
                                   size=(IMAGE_HEIGHT, IMAGE_LENGTH)).astype(int)
    white_noise_more = np.random.normal(loc=0, scale=float(sqrt(avg_dispersion / WHITE_NOISE_MORE_PARAMETER)),
                                        size=(IMAGE_HEIGHT, IMAGE_LENGTH)).astype(int)
    #  dispersion_noise = np.var(white_noise)
    correct1 = np.abs(white_noise.min())
    correct2 = np.abs(white_noise_more.min())

    noise_to_show1 = white_noise + correct1
    noise_to_show1 = check_and_correct_limits(noise_to_show1)

    noise_to_show2 = white_noise_more + correct2
    noise_to_show2 = check_and_correct_limits(noise_to_show2)

    noise_img_with_T_random = img_with_T_random_positions + white_noise_more
    noise_img_with_T_random = check_and_correct_limits(noise_img_with_T_random)

    noise_img_with_T_and_L_random = img_with_T_and_L_random + white_noise
    noise_img_with_T_and_L_random = check_and_correct_limits(noise_img_with_T_and_L_random)

    show_source_imgs(noise_to_show2, noise_to_show1, img_with_T_random_positions, img_with_T_and_L_random)
    show()
    show_source_imgs(noise_to_show2, noise_to_show1, noise_img_with_T_random, noise_img_with_T_and_L_random)
    show()

    correlate_field_T_model1 = Correlator(img_with_T_random_positions, MODEL1)
    correlate_field_TandL_model1 = Correlator(img_with_T_and_L_random, MODEL1)
    correlate_field_TandL_model2 = Correlator(img_with_T_and_L_random, MODEL2)

    correlate_field_noise_T_model1_random = Correlator(noise_img_with_T_random, MODEL1)
    correlate_field_noise_TandL_model1_random = Correlator(noise_img_with_T_and_L_random, MODEL1)
    correlate_field_noise_TandL_model2_random = Correlator(noise_img_with_T_and_L_random, MODEL2)

    max_in_T_random = correlate_field_T_model1.max()
    max_in_L_random = correlate_field_TandL_model2.max()

    conrast1 = lineary_contrast(correlate_field_T_model1, max_in_T_random)
    conrast2 = lineary_contrast(correlate_field_TandL_model1, max_in_T_random)
    conrast3 = lineary_contrast(correlate_field_TandL_model2, max_in_T_random)

    noise_conrast1 = lineary_contrast(correlate_field_noise_T_model1_random, max_in_T_random)
    noise_conrast2 = lineary_contrast(correlate_field_noise_TandL_model1_random, max_in_T_random)
    noise_conrast3 = lineary_contrast(correlate_field_noise_TandL_model2_random, max_in_T_random)

    border1 = border_processing(conrast1, 200)
    border2 = border_processing(conrast2, 170)
    border3 = border_processing(conrast3, 170)

    noise_border1 = border_processing(noise_conrast1, 230)
    noise_border2 = border_processing(noise_conrast2, 190)
    noise_border3 = border_processing(noise_conrast3, 190)

    show_fields(img_with_T_random_positions, correlate_field_T_model1, border1,
                noise_img_with_T_random, correlate_field_noise_T_model1_random, noise_border1, 'model T')
    show()
    show_fields(img_with_T_and_L_random, correlate_field_TandL_model1, border2,
                noise_img_with_T_and_L_random, correlate_field_noise_TandL_model1_random, noise_border2, 'model T')
    show()
    show_fields(img_with_T_and_L_random, correlate_field_TandL_model2, border3,
                noise_img_with_T_and_L_random, correlate_field_noise_TandL_model2_random, noise_border3, 'model INVERSE_T')
    show()


















