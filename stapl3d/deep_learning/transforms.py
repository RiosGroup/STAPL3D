import random
from copy import copy

import numpy as np
from scipy.ndimage import rotate
from skimage.transform import rescale
from skimage.util import random_noise
import elasticdeform

class Augmenter():
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, data):
        return (self.transform(data))

    def transform(self, data):
        for aug in self.augmentations:
            data = aug(data)
        return (data)


class Augmentation():
    def __init__(self, p=1.0):
        assert 0 <= p <= 1
        self.p = p

    def __call__(self, data):
        """
        Expects data to be {"image":img, "mask":mask}
        """
        data = copy(data)
        if random.uniform(0, 1) <= self.p:
            self.params = self.get_params()
            if isinstance(data, np.ndarray):
                data = self.apply(data, **self.params)
            elif isinstance(data, dict):
                if "image" in data.keys():
                    data["image"] = self.apply(data["image"], **self.params)
                if "mask" in data.keys():
                    data["mask"] = self.apply_to_mask(data["mask"], **self.params)
            else:
                print(f"Unknown data type: {type(data)}")
        return (data)

    def get_params(self):
        """
        shared parameters for one apply. (usually random values)
        """
        return {}

    def apply_to_mask(self, img, **params):
        """
        shared parameters for one apply. (usually random values)
        """
        return img


class Normalize(Augmentation):
    """
    Z-score normalization
    """

    def __init__(self, pixel_range=(), p=1.0):
        super().__init__(p)
        self.pixel_range = pixel_range

    def apply(self, image):
        if self.pixel_range:
            mn = image.min()
            mx = image.max()
            image = (image - mn) / (mx - mn)
        mean = image.mean()
        std = image.std()
        denominator = np.reciprocal(std)
        image = (image - mean) * denominator
        return image


class BrightnessNoise(Augmentation):
    def __init__(self, val_range=(0, 0.5), clip=True, p=1.0):
        super().__init__(p)
        self.val_range = val_range
        self.clip = clip

    def apply(self, image, brightness=150):
        np_bright = np.full(image.shape, brightness)
        image = image + np_bright
        if self.clip:
            image = np.clip(image, 0.0, 1.0)
        return (image)

    def get_params(self):
        brightness = random.uniform(self.val_range[0], self.val_range[1])
        return {
            "brightness": brightness
        }


class GaussianNoise(Augmentation):  # pixel_proportion= 0.05,
    """
    data is form {"image": image, "mask": mask}
    """

    def __init__(self, mean=0.0, var_range=(0, 0.001), clip=True, p=1.0):
        super().__init__(p)
        self.mean = mean
        self.var_range = var_range
        self.clip = clip

    def apply(self, image, var=0.001):
        image = random_noise(image, mode='gaussian', mean=self.mean, var=var, clip=False)
        if self.clip:
            image = np.clip(image, 0.0, 1.0)
        return (image)

    def get_params(self):
        var = random.uniform(self.var_range[0], self.var_range[1])
        return {
            "var": var
        }


class CenterCrop(Augmentation):
    def __init__(self, crop_shape=(32, 320, 320), p=1.0):
        super().__init__(p)
        self.crop_shape = crop_shape

    def apply(self, image, crop_perc=[0.5, 0.5, 0.5]):
        img_shape = image.shape
        shape_diff = len(img_shape) - len(self.crop_shape)
        if shape_diff > 0:
            self.crop_shape = img_shape[0:shape_diff] + self.crop_shape
        self.crop_coords = []
        for idx, ax in enumerate(self.crop_shape):
            assert ax <= img_shape[
                idx], f"{ax} in crop shape {self.crop_shape} is larger than original image {img_shape}"
            if ax == img_shape[idx]:
                self.crop_coords.append(slice(0, ax))
            else:
                self.test = []
                max_start = img_shape[idx] - ax
                self.test = self.test + [max_start]
                start = int(round(0.5 * max_start))
                self.test = self.test + [start]
                end = start + ax
                self.test = self.test + [end]
                self.crop_coords.append(slice(start, end))

        image = image[tuple(self.crop_coords)]
        return (image)

    def apply_to_mask(self, image, crop_perc=[0.5, 0.5, 0.5]):
        return self.apply(image, crop_perc)

    def get_params(self):
        crop_perc = [random.random() for i in self.crop_shape]
        return {
            "crop_perc": crop_perc
        }

class RandomCrop(Augmentation):
    def __init__(self, crop_shape=(32, 320, 320), p=1.0):
        super().__init__(p)
        self.crop_shape = crop_shape

    def apply(self, image, crop_perc):
        img_shape = image.shape
        crop_shape =  self.crop_shape
        shape_diff = len(img_shape) - len(crop_shape)
        if shape_diff > 0:
            crop_shape = img_shape[0:shape_diff] + crop_shape
            crop_perc = [0]*shape_diff + crop_perc
        self.crop_coords = []
        for idx, ax in enumerate(crop_shape):
            assert ax <= img_shape[
                idx], f"{ax} in crop shape {crop_shape} is larger than original image {img_shape}"
            if ax == img_shape[idx]:
                self.crop_coords.append(slice(0, ax))
            else:
                self.test = []
                max_start = img_shape[idx] - ax
                self.test = self.test + [max_start]
                start = int(round(crop_perc[idx] * max_start))
                self.test = self.test + [start]
                end = start + ax
                self.test = self.test + [end]
                self.crop_coords.append(slice(start, end))
        image = image[tuple(self.crop_coords)]
        return (image)

    def apply_to_mask(self, image, crop_perc):
        return self.apply(image, crop_perc)

    def get_params(self):
        crop_perc = [random.random() for i in self.crop_shape]
        return {
            "crop_perc": crop_perc
        }

class Flip(Augmentation):
    def __init__(self, axis, p=1.0):
        super().__init__(p)
        self.axis = axis

    def apply(self, image):
        if isinstance(self.axis, str):
            if self.axis == "x":
                ax = -1
            elif self.axis == "y":
                ax = -2
            elif self.axis == "z":
                ax = -3
            else:
                raise (f"{self.axis} is not a known axis name. Please provide x,y,z or a number")
        else:
            ax = self.axis
        image = np.flip(image, axis=ax).copy()
        return (image)

    def apply_to_mask(self, image):
        return self.apply(image)


class Rotate(Augmentation):
    def __init__(
            self,
            angle_range=(-15, 15),
            axes=(-1, -2),
            reshape=False,
            interpolation=3,
            border_mode='constant',
            value=0,
            clip=True,
            p=1.0
    ):

        super().__init__(p)
        self.angle_range = angle_range
        if isinstance(axes, str):
            if axes == "xy" or axes == "yx":
                self.axes = (-1, -2)
            elif axes == "yz" or axes == "zy":
                self.axes = (-2, -3)
            elif axes == "xz" or axes == "zx":
                self.axes = (-1, -3)
            else:
                warn("Defined axes name {axes} not a valid type (e.g. xy, yx, zy)")
        else:
            self.axes = axes
        self.reshape = reshape
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.clip = clip

    def apply(self, image, angle=15):
        image = rotate(image, angle, axes=self.axes, reshape=self.reshape, order=self.interpolation,
                       mode=self.border_mode, cval=self.value)
        if self.clip:
            image = np.clip(image, 0.0, 1.0)
        return image

    def apply_to_mask(self, image, angle=15):
        return rotate(image, angle, axes=self.axes, reshape=self.reshape, order=0, mode=self.border_mode,
                      cval=self.value)

    def get_params(self):
        assert len(self.angle_range) == 2
        angle = random.uniform(self.angle_range[0], self.angle_range[1])
        return {
            "angle": angle
        }


class SaltPepperNoise(Augmentation):
    def __init__(self, amount_range=(0, 0.05), salt_vs_pepper_range=(0.4, 0.6), p=1.0):
        super().__init__(p)
        self.salt_vs_pepper_range = salt_vs_pepper_range
        self.amount_range = amount_range

    def apply(self, image, amount=0.05, salt_vs_pepper=0.5):
        image = random_noise(image, mode='s&p', amount=amount, salt_vs_pepper=salt_vs_pepper)
        return image

    def get_params(self):
        amount = random.uniform(self.amount_range[0], self.amount_range[1])
        salt_vs_pepper = random.uniform(self.salt_vs_pepper_range[0], self.salt_vs_pepper_range[1])
        return {
            "amount": amount,
            "salt_vs_pepper": salt_vs_pepper
        }

class Scale(Augmentation):
    def __init__(self, scale_x_range=(0.5, 1.5), scale_y_range=(0.5, 1.5), p=1.0):
        super().__init__(p)
        self.scale_x_range = scale_x_range
        self.scale_y_range = scale_y_range

    def apply(self, image, scale_x, scale_y):
        print((scale_x, scale_y))
        print(image.shape)
        image = rescale(image, scale=(1, scale_y, scale_x), clip=False)  # scaling is (z,y,x)
        print(image.shape)
        return image

    def get_params(self):
        scale_x = random.uniform(self.scale_x_range[0], self.scale_x_range[1])
        scale_y = random.uniform(self.scale_y_range[0], self.scale_y_range[1])
        return {
            "scale_x": scale_x,
            "scale_y": scale_y
        }

class ElasticDeformation(Augmentation):
    def _init_(self, sigma_range=(-10,10), points_range=(2,4), p=1.0):
        super()._init_(p)
        self.sigma_range = sigma_range
        self.points_range = points_range
        self.displacement = None


    def apply(self, image, sigma=25, points=3, mask=False):
        # some messy logic to ensure the same deformation on the mask as on the image
        if not mask:
            Xs = self._normalize_inputs(image)
            _, deform_shape = self._normalize_axis_list(None, Xs)
            if not isinstance(points, (list, tuple)):
                points = [points] * len(deform_shape)
            self.displacement = np.random.randn(len(deform_shape), *points) * sigma
            image = elasticdeform.deform_grid(image, self.displacement)
            image = np.clip(image, 0.0, 1.0)
        elif mask:
            # assuming mask is of type bool or int
            # TODO: try to scale values to range 0.0 - 1.0 in stead of clipping
            image = np.array(image, dtype=float)
            image = elasticdeform.deform_grid(image, self.displacement)
            image = np.around(image, decimals=0)
            image = np.clip(image, 0.0, 1.0)
            image = np.array(image, dtype=bool)
        return image


    def apply_to_mask(self, image, sigma=25, points=3, mask=True):
        return self.apply(image, sigma, points, mask=True)


    def get_params(self):
        sigma = random.randint(self.sigma_range[0], self.sigma_range[1])
        points = random.randint(self.points_range[0], self.points_range[1])
        return{
            "sigma": sigma,
            "points": points
        }


    def _normalize_inputs(self, X):
    # adapted from elastic deform source code
        if isinstance(X, np.ndarray):
            Xs = [X]
        elif isinstance(X, list):
            Xs = X
        else:
            raise Exception('X should be a numpy.ndarray or a list of numpy.ndarrays.')

        # check X inputs
        assert len(Xs) > 0, 'You must provide at least one image.'
        assert all(isinstance(x, np.ndarray) for x in Xs), 'All elements of X should be numpy.ndarrays.'
        return Xs


    def _normalize_axis_list(self, axis, Xs):
    # adapted from elastic deform source code
        if axis is None:
            axis = [tuple(range(x.ndim)) for x in Xs]
        elif isinstance(axis, int):
            axis = (axis,)
        if isinstance(axis, tuple):
            axis = [axis] * len(Xs)
        assert len(axis) == len(Xs), 'Number of axis tuples should match number of inputs.'
        input_shapes = []
        for x, ax in zip(Xs, axis):
            assert isinstance(ax, tuple), 'axis should be given as a tuple'
            assert all(isinstance(a, int) for a in ax), 'axis must contain ints'
            assert len(ax) == len(axis[0]), 'All axis tuples should have the same length.'
            assert ax == tuple(set(ax)), 'axis must be sorted and unique'
            assert all(0 <= a < x.ndim for a in ax), 'invalid axis for input'
            input_shapes.append(tuple(x.shape[d] for d in ax))
        assert len(set(input_shapes)) == 1, 'All inputs should have the same shape.'
        deform_shape = input_shapes[0]
        return axis, deform_shape

def test():
    from general_segmentation_functions.image_handling import Image, get_image, view_napari
    im=Image("/Users/samdeblank/Documents/1.projects/brain_segmentation/coregistration/coregistered_h5/20210416_MBR26_B16_zstack_pair1.h5/", permission="r")

    img=im.load(substruct="/dapi/raw_25x")
    lbl=im.load(substruct="/dapi/labels")
    input_data={"image":img, "mask":lbl}
    aug_substruct="/augmented/"
    # img = img / 65535.0
    augmenter = Augmenter(
        [
    # Flip(axis="x", p=0.5),
    # Flip(axis="y", p=0.5),
    # Flip(axis="z", p=0.5),
    # Rotate(angle_range=(-15,15), axes=(-1,-2), p=1.0),
    # RandomCrop((32,320,320), p=1.0),
    RandomCrop((1,200,200), p=1.0),
    # GaussianNoise(var_range=(0, 0.003), p=1.0),
    # BrightnessNoise(val_range=(-0.2,-0.2), clip=True, p=1.0),
    # SaltPepperNoise(amount_range=(0.33, 0.34), salt_vs_pepper_range=(0.0, 0.0), p=1.0),
    # Scale(scale_x_range=(0.5, 1.5), scale_y_range=(0.5, 1.5), p=1.0)
    ]
    )
    data=augmenter(input_data)

    # data=Rotate3D(data, xy_range=(0,0), yz_range=(0,0), xz_range=(-15,15), interpolation=3, border_mode='constant', value=0)
    im.write(img, substruct="original_image", outpath="/Users/samdeblank/Downloads/test.h5")
    im.write(data["image"], substruct="augmented_image3", outpath="/Users/samdeblank/Downloads/test.h5")
    # im.write(data["mask"], substruct="augmented_mask")
    im.close()

    img=get_image("/Users/samdeblank/Documents/1.projects/BEHAV3D_HT/1.data/toy_datasets/single_fast_vs_static_classification_toyset/train/images/00008.tif")
    input_data={"image":img}
    augmenter = Augmenter(
        [
    Flip(axis="x", p=1.0),
    Flip(axis="y", p=1.0),
    # Flip(axis="z", p=0.5),
    # Rotate(angle_range=(-15,15), axes=(-1,-2), p=1.0),
    # RandomCrop((32,320,320), p=1.0),
    # RandomCrop((1,200,200), p=1.0),
    GaussianNoise(var_range=(0, 0.003), p=1.0),
    BrightnessNoise(val_range=(-0.2,-0.2), clip=True, p=1.0),
    # SaltPepperNoise(amount_range=(0.33, 0.34), salt_vs_pepper_range=(0.0, 0.0), p=1.0),
    # Scale(scale_x_range=(0.5, 1.5), scale_y_range=(0.5, 1.5), p=1.0)
    ]
    )
    data=augmenter(input_data)["image"]
    view_napari([img, data])