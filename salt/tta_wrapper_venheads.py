
# coding: utf-8

# In[7]:


import tensorflow as tf
import itertools
from keras.layers import Layer
from keras.models import Model
from keras.layers import Input
import Funct_wrapper as F
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"




# In[4]:


def add_identity(params):
    """Add identical parameter (0) for all manipulations"""
    if params is None:
        res = [0]
    elif isinstance(params, bool):
        if params:
            res = [0] + [int(params)]
        else:
            res = [0]
    elif isinstance(params, tuple):
        res = [0] + list(params)
    elif isinstance(params, list):
        res = [0] + params
    else:
        raise ValueError('Wrong param type')
    return res


def invert(params):
    """Invert order of parameters for manipulations"""
    return list(map(lambda x: -x, params))


class Augmentation(object):

    def __init__(self,
                 h_flip=True,
                 v_flip=True,
                 h_shifts=(10, -10),
                 v_shifts=(10, -10),
                 rotation_angles=(90, 180, 270),):

        super().__init__()

        self.h_flip = add_identity(h_flip)
        self.v_flip = add_identity(v_flip)
        self.rotation_angles = add_identity(rotation_angles)
        self.h_shifts = add_identity(h_shifts)
        self.v_shifts = add_identity(v_shifts)

        self.n_transforms = len(self.h_flip) *                             len(self.v_flip)*                             len(self.rotation_angles) *                             len(self.h_shifts) *                             len(self.v_shifts)

        self.forward_aug = [F.h_flip, F.v_flip, F.rotate, F.h_shift, F.v_shift]
        self.forward_transform_params = list(itertools.product(
            self.h_flip,
            self.v_flip,
            self.rotation_angles,
            self.h_shifts,
            self.v_shifts,
        ))

        self.backward_aug = self.forward_aug[::-1]

        backward_transform_params = list(itertools.product(
            invert(self.h_flip),
            invert(self.v_flip),
            invert(self.rotation_angles),
            invert(self.h_shifts),
            invert(self.v_shifts),
        ))
        self.backward_transform_params = list(map(lambda x: x[::-1],
                                        backward_transform_params))

    @property
    def forward(self):
        return self.forward_aug, self.forward_transform_params

    @property
    def backward(self):
        return self.backward_aug, self.backward_transform_params


# In[6]:


class Repeat(Layer):
    """
    Layer for cloning input information
    input_shape = (1, H, W, C)
    output_shape = (N, H, W, C)
    """
    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def call(self, x):
        return tf.stack([x[0]] * self.n, axis=0)

    def compute_output_shape(self, input_shape):
        return (self.n, *input_shape[1:])


class TTA(Layer):

    def __init__(self, functions, params):
        super().__init__()
        self.functions = functions
        self.params = params

    def apply_transforms(self, images):
        transformed_images = []
        for i, args in enumerate(self.params):
            image = images[i]
            for f, arg in zip(self.functions, args):
                image = f(image, arg)
            transformed_images.append(image)
        return tf.stack(transformed_images, 0)

    def call(self, images):
        return self.apply_transforms(images)


class Merge(Layer):

    def __init__(self, type):
        super().__init__()
        self.type = type

    def merge(self, x):
        if self.type == 'mean':
            return F.mean(x)
        if self.type == 'gmean':
            return F.gmean(x)
        if self.type == 'max':
            return F.max(x)
        else:
            raise ValueError(f'Wrong merge type {type}')

    def call(self, x):
        return self.merge(x)

    def compute_output_shape(self, input_shape):
        return (1, *input_shape[1:])


# In[8]:


doc = """
    IMPORTANT constraints:
        1) model has to have 1 input and 1 output
        2) inference batch_size = 1
        3) image height == width if rotate augmentation is used
    Args:
        model: instance of Keras model
        h_flip: (bool) horizontal flip
        v_flip: (bool) vertical flip
        h_shifts: (list of int) list of horizontal shifts (e.g. [10, -10])
        v_shifts: (list of int) list of vertical shifts (e.g. [10, -10])
        rotation_angles: (list of int) list of angles (deg) for rotation in range [0, 360),
            should be divisible by 90 deg (e.g. [90, 180, 270])
        merge: one of 'mean', 'gmean' and 'max' - mode of merging augmented
            predictions together.
    Returns:
        Keras Model instance
"""

def tta_segmentation(model,
                     h_flip=False,
                     v_flip=False,
                     h_shifts=None,
                     v_shifts=None,
                     rotation_angles=None,
                     merge='mean'):

    """
    Segmentation model test time augmentation wrapper.
    """
    tta = Augmentation(h_flip=h_flip,
                       v_flip=v_flip,
                       h_shifts=h_shifts,
                       v_shifts=v_shifts,
                       rotation_angles=rotation_angles)

    input_shape = (1, *model.get_input_at(0).shape.as_list()[1:])

    inp = Input(batch_shape=input_shape)
    x = Repeat(tta.n_transforms)(inp)
    x = TTA(*tta.forward)(x)
    x = model(x)
    x = TTA(*tta.backward)(x)
    x = Merge(merge)(x)
    tta_model = Model(inp, x)

    return tta_model


def tta_classification(model,
                       h_flip=False,
                       v_flip=False,
                       h_shifts=None,
                       v_shifts=None,
                       rotation_angles=None,
                       merge='mean'):
    """
    Classification model test time augmentation wrapper.
    """

    tta = Augmentation(h_flip=h_flip,
                       v_flip=v_flip,
                       h_shifts=h_shifts,
                       v_shifts=v_shifts,
                       rotation_angles=rotation_angles)

    input_shape = (1, *model.get_input_at(0).shape.as_list()[1:])

    inp = Input(batch_shape=input_shape)
    x = Repeat(tta.n_transforms)(inp)
    x = TTA(*tta.forward)(x)
    x = model(x)
    x = Merge(merge)(x)
    tta_model = Model(inp, x)

    return tta_model


tta_classification.__doc__ += doc
tta_segmentation.__doc__ += doc

