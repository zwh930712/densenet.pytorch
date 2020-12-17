import numbers
import numpy as np
import random
import cv2


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    box_a = box_a[np.newaxis, :]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

    union = area_a + area_b - inter
    return inter / union


def box_crop(img, box):
    """box ordered [y1, x1, y2, x2]"""
    crop_widths = box[3] - box[1]
    crop_heights = box[2] - box[0]
    if crop_widths < 32:
        box[3] = np.clip(box[3] + int((32 - crop_widths) / 2.0), 0, img.shape[0])
        box[1] = np.clip(box[1] - int((32 - crop_widths) / 2.0), 0, img.shape[0])
    if crop_heights < 32:
        box[2] = np.clip(box[2] + int((32 - crop_heights) / 2.0), 0, img.shape[1])
        box[0] = np.clip(box[0] - int((32 - crop_heights) / 2.0), 0, img.shape[1])
    crop_img = img[box[1]:box[3], box[0]:box[2], :]
    return crop_img


def random_global_crop(img, imgsz, crop_size):
    crop_widths = int(random.uniform(crop_size[0], crop_size[0] * 3))
    crop_heights = int(random.uniform(crop_size[1], crop_size[1] * 3))
    crop_widths = np.minimum(crop_widths, imgsz[0])
    crop_heights = np.minimum(crop_heights, imgsz[1])

    x1 = int(random.randint(0, imgsz[0] - crop_widths - 1))
    y1 = int(random.randint(0, imgsz[1] - crop_heights - 1))

    x2 = int(x1 + crop_heights)
    y2 = int(y1 + crop_widths)

    return box_crop(img, [y1, x1, y2, x2])


def random_box_crop(img, imgsz, box, num_attempts=50):
    """box ordered [y1, x1, y2, x2]"""
    scale = (0.5, 1.0)
    min_iou = 0.25
    box_width = box[3] - box[1]
    box_height = box[2] - box[0]
    while True:
        for _ in range(num_attempts):

            w = random.uniform(scale[0] * box_width, scale[1] * box_width)
            h = random.uniform(scale[1] * box_height, scale[1] * box_height)

            x1 = np.clip(box[1] + w, 0, imgsz[0])
            y1 = np.clip(box[0] + h, 0, imgsz[1])
            x2 = np.clip(box[3] + w, 0, imgsz[0])
            y2 = np.clip(box[2] + h, 0, imgsz[1])

            # convert to inter rect x1, y1, x2, y2
            rect = np.array([int(y1), int(x1), int(y2), int(x2)])

            # calculate Iou (jaccard overlap) b /t the cropped and gt boxes
            overlap = jaccard_numpy(np.array(box), rect)

            # is min and max overlap constraint satisfied ? if not try again
            if overlap.min() < min_iou:
                return box_crop(img, rect)

        return random_global_crop(img, imgsz, crop_size=(64, 64))


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        try:
            image = cv2.resize(image, (self.size[0], self.size[1]))
        except:
            print(image.shape)
        return image


class Compose(object):
    """Composes several augmentations together.
       Args:
           transforms (List[Transform]): list of transforms to compose.
       Example:
           >>> augmentations.Compose([
           >>>     transforms.CenterCrop(10),
           >>>     transforms.ToTensor(),
           >>> ])
       """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class CenterCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.shape[0], img.shape[1]
        th, tw = self.size

        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))

        if x1 > 0 and y1 > 0:
            return box_crop(img, (y1, x1, y1 + th, x1 + tw))
        else:
            return img


class NormLize:
    def __call__(self, img):
        return np.array(img / 255., dtype=np.float32)
        # return np.array(img / 255., dtype=np.float32).transpose(2, 0, 1)
