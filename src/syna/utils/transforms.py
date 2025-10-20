import numpy as np
from PIL import Image

from syna import utils


class Compose:
    """
    Compose a sequence of image transforms.

    Applies each transform in order to the input image and returns the result.
    The transforms can accept and return either PIL Images or numpy arrays,
    depending on the transform implementation.
    """

    def __init__(self, transforms=None):
        self.transforms = transforms or []

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Convert:
    """
    Convert a PIL Image to the requested color mode.

    The transform accepts a PIL Image and returns another PIL Image converted
    to `mode`. A special-case mode "BGR" converts the image from RGB to BGR
    by swapping channels (useful for interoperability with frameworks expecting
    BGR order).
    """

    def __init__(self, mode="RGB"):
        self.mode = mode

    def __call__(self, img):
        if self.mode == "BGR":
            img = img.convert("RGB")
            r, g, b = img.split()
            return Image.merge("RGB", (b, g, r))
        return img.convert(self.mode)


class Resize:
    """
    Resize a PIL Image to the given size.

    Size may be an int or a tuple. The `pair` helper ensures size is (width, height).
    The `interpolation` argument accepts PIL resampling filters (default BILINEAR).
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = utils.pair(size)
        # named 'interpolation' to match torchvision.Resize
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize(self.size, self.interpolation)


class CenterCrop:
    """
    Crop the center region of a PIL Image.

    The `size` may be an int or (width, height). If the requested size is larger
    than the image, behavior follows PIL.crop (it will include padding from image).
    """

    def __init__(self, size):
        self.ow, self.oh = utils.pair(size)

    def __call__(self, img):
        w, h = img.size
        left = (w - self.ow) // 2
        upper = (h - self.oh) // 2
        return img.crop((left, upper, left + self.ow, upper + self.oh))


class ToTensor:
    """
    Convert a PIL Image or numpy.ndarray to a float32 numpy array.

    Output shape is (C, H, W). For uint8 input the values are scaled to [0.0, 1.0]
    to match torchvision.transforms.ToTensor. Supports grayscale and RGB images.
    """

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # avoid copying if already an ndarray
            arr = np.asarray(pic)
            # H x W -> H x W x 1
            if arr.ndim == 2:
                arr = arr[:, :, None]
            # H x W x C -> C x H x W
            if arr.ndim == 3:
                arr = arr.transpose(2, 0, 1)
            if arr.dtype == np.uint8:
                # prefer astype(copy=False) to avoid allocation when dtype already matches
                return arr.astype(np.float32, copy=False) / 255.0
            return arr.astype(np.float32, copy=False)

        if isinstance(pic, Image.Image):
            # get array view of PIL Image (no extra copy)
            arr = np.asarray(pic)
            if arr.ndim == 2:
                arr = arr[:, :, None]
            arr = arr.transpose(2, 0, 1)
            return arr.astype(np.float32, copy=False) / 255.0

        raise TypeError("ToTensor expected PIL.Image or numpy.ndarray")


class ToPILImage:
    """
    Convert a numpy array to a PIL Image.

    Accepts arrays in shape (C, H, W), (H, W, C) or (H, W). Float images in
    [0, 1] will be scaled to [0, 255] and cast to uint8. Other dtypes are
    converted to uint8.
    """

    def __call__(self, array):
        arr = np.asarray(array)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            # C x H x W -> H x W x C
            arr = arr.transpose(1, 2, 0)
        elif arr.ndim == 3 and arr.shape[2] in (1, 3):
            # already H x W x C
            pass
        elif arr.ndim == 2:
            # H x W grayscale
            pass
        else:
            raise TypeError(
                "ToPILImage expected array of shape (C,H,W), (H,W,C) or (H,W)"
            )

        # For float data, if values are in [0,1] scale to [0,255].
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0:
                arr = (arr * 255.0).round()
            arr = arr.astype(np.uint8, copy=False)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)

        # Convert single-channel H x W x 1 to H x W for PIL
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]

        return Image.fromarray(arr)


class RandomHorizontalFlip:
    """
    Randomly horizontally flip an image with probability p.

    Works with both PIL Images and numpy arrays. For numpy arrays this flips
    along the last horizontal axis and returns a copy to preserve contiguity.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            if isinstance(img, Image.Image):
                return img.transpose(Image.FLIP_LEFT_RIGHT)
            if isinstance(img, np.ndarray):
                # Works for (C,H,W), (H,W,C) and (H,W)
                # return a copy to ensure contiguity and avoid surprising views
                return img[..., ::-1].copy()
        return img


class Normalize:
    """
    Normalize a (C, H, W) numpy array by mean and std.

    Mean and std may be scalars or sequences of length C. Broadcasts to match
    the input array shape and returns a float array with (arr - mean) / std.
    """

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, array):
        arr = array
        mean, std = self.mean, self.std
        if not np.isscalar(mean):
            mean = np.array(mean, dtype=arr.dtype).reshape(
                (len(mean),) + (1,) * (arr.ndim - 1)
            )
        if not np.isscalar(std):
            std = np.array(std, dtype=arr.dtype).reshape(
                (len(std),) + (1,) * (arr.ndim - 1)
            )
        return (arr - mean) / std


class Flatten:
    """
    Flatten an array to 1-D.

    Returns a flattened view where possible (ravel), suitable for converting
    image tensors to vectors.
    """

    def __call__(self, array):
        # prefer ravel (view) to minimize memory where possible
        return array.ravel()


class ConvertImageDtype:
    """
    Cast a numpy image array to the specified dtype.

    This is a simple dtype conversion utility similar in intent to
    torchvision.transforms.ConvertImageDtype but implemented for numpy arrays.
    """

    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, array):
        # prefer copy=False to avoid allocating when dtype already matches
        return array.astype(self.dtype, copy=False)


# compatibility aliases matching torchvision names where applicable
ToFloat = ConvertImageDtype


def ToInt(dtype=int):
    return ConvertImageDtype(dtype)
