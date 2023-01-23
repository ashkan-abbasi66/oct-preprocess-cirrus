import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arrow, Circle



def show_sample(img_vol,
                slice=100,  # B-scan number
                title = ''):
    fig = plt.figure()

    plt.subplot(1, 2, 1)
    plt.imshow(img_vol[:, :, slice])

    plt.subplot(1, 2, 2)
    enface = np.mean(img_vol, axis=0)
    plt.imshow(enface)

    fig.suptitle(title)


def show_image(image,title="",show=True):
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    if show:
        plt.show()

def mark_max_location(input_image,overlay_on = None,show=True):
    if overlay_on is None:
        overlay_on = input_image

    image = input_image.copy()
    maxloc = np.argmax(image)
    print(maxloc)

    # convert linear index to a 2D coordinate
    x0, y0 = np.unravel_index(maxloc, image.shape, 'C')

    fig, ax = plt.subplots(1)
    # plt.imshow(image)
    plt.imshow(overlay_on)
    ax.add_patch(Circle((x0, y0), radius=2, color='red'))
    if show is True:
        plt.show()
    return x0, y0