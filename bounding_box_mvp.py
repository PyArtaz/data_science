import numpy as np
import cv2
from util_mvp import find_min_max


class BoundingBox:
    def __init__(self, landmarks, img_width, img_height, margin=10):
        """
        Create a bounding box around the hand(s) passed as landmark(s). The same coordinate system as in the input
        landmarks is used.

        Parameters
        ----------
        landmarks : ndarray
            The landmark coordinates of the hand in the image. Shape n x 63.
        img_width : int, ndarray
            The image width in pixels.
        img_height : int, ndarray
            The image height in pixels.
        margin : int
            The amount of pixels that the bounding box would be larger on the image. The default is 10.

        Returns
        -------

        """
        coordinates = landmarks.reshape(-1, 21, 3)
        img_width = np.asarray([img_width]) if np.isscalar(img_width) else np.asarray(img_width)
        img_height = np.asarray([img_height]) if np.isscalar(img_height) else np.asarray(img_height)

        self.aspect_ratio = img_width/img_height

        self.x_bound = find_min_max(coordinates[:, :, 0])
        self.y_bound = find_min_max(coordinates[:, :, 1])

        x_margin = (margin / img_width).reshape(-1, 1)
        y_margin = (margin / img_height).reshape(-1, 1)

        self.x_bound += np.concatenate((-x_margin, x_margin), axis=1)
        self.y_bound += np.concatenate((-y_margin, y_margin), axis=1)

    def draw(self, image):
        """
        Draw a bounding box containing the hand on a given image.

        Parameters
        ----------
        image : ndarray
            The image to draw the bounding box on.

        Returns
        -------
        ndarray
            The modified image including the bounding box.

        """
        img = image
        img_height, img_width, _ = img.shape

        x_bounds = np.rint(self.x_bound * img_width).astype(int)
        y_bounds = np.rint(self.y_bound * img_height).astype(int)

        x_bounds, y_bounds = enforce_bounds(x_bounds, y_bounds, img_width, img_height)

        cv2.rectangle(img, (x_bounds[0, 0], y_bounds[0, 0]), (x_bounds[0, 1], y_bounds[0, 1]), (0, 0, 0), 2)

        return img

    def make_square(self):
        """
        Make an existing bounding box square in pixels such that the whole hand fits inside.

        Parameters
        ----------

        Returns
        -------

        """
        aspect_ratio = np.ones((self.x_bound.shape[0], 1)) * self.aspect_ratio.reshape(-1, 1)
        diff_dist = np.diff(self.x_bound) - np.diff(self.y_bound) / aspect_ratio

        for i, dist in enumerate(diff_dist):
            if dist < 0:
                self.x_bound[i, :] += np.array([[dist.item() / 2, -dist.item() / 2]]).reshape(-1)
            elif dist > 0:
                self.y_bound[i, :] += np.array(
                    [[-dist.item() * aspect_ratio[i] / 2, dist.item() * aspect_ratio[i] / 2]]).reshape(-1)

    def coordinates_to_bb(self, coordinates):
        """
        Calculate the coordinates relative to the bounding box. The origin will be in the upper left corner of
        the bounding box, whereas the point (1, 1) will be in the lower left corner.

        Parameters
        ----------
        coordinates : ndarray
            The landmark coordinates of the hand in the image. Shape n x 63.

        Returns
        -------
        ndarray
            The shifted coordinates as described above. Same shape as coordinates (n x 63).

        """
        new_origin = np.zeros((self.x_bound.shape[0], 3))
        new_origin[:, 0] = self.x_bound[:, 0]
        new_origin[:, 1] = self.y_bound[:, 0]
        new_origin_tile = np.tile(new_origin, 21)
        coo_shift = coordinates.copy()
        coo_shift -= new_origin_tile

        x_range = np.diff(self.x_bound)
        y_range = np.diff(self.y_bound)
        z_range = (x_range + y_range) / 2

        coo_shift[:, 0:61:3] /= x_range.reshape((-1, 1))
        coo_shift[:, 1:62:3] /= y_range.reshape((-1, 1))
        coo_shift[:, 2:63:3] /= z_range.reshape((-1, 1))

        return coo_shift


def enforce_bounds(x_border, y_border, img_width, img_height):
    """
    Ensure that the bounding box lies within in the image. If any coordinates don't, they will be transferred to the
    nearest point within the image.

    Parameters
    ----------
    x_border : ndarray
        The minimum and maximum x-coordinates in the coordinates of one hand. Shape n x 2.
    y_border : ndarray
        The minimum and maximum y-coordinates in the coordinates of one hand. Shape n x 2.
    img_width : int
        The image width in pixels.
    img_height: int
        The image height in pixels.

    Returns
    -------
    x_border : ndarray
        The modified x-limits such that all coordinates lie within the image. Shape n x 2.
    y_border : ndarray
        The modified x-limits such that all coordinates lie within the image. Shape n x 2.

    """
    if x_border[0, 0] < 0:
        x_border[0, 0] = 0
    if y_border[0, 0] < 0:
        y_border[0, 0] = 0
    if x_border[0, 1] > img_width:
        x_border[0, 1] = img_width
    if y_border[0, 1] > img_height:
        y_border[0, 1] = img_height

    return x_border, y_border
