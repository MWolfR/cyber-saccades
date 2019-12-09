from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf


class Saccades(Layer):
    """An attention system simulating saccades as a Keras layer.

    Notes
    -----


    """

    def __init__(self, output_width=400, output_height=300, **kwargs):
        super(Saccades, self).__init__(**kwargs)
        self._tgt_w = output_width
        self._tgt_h = output_height

    def compute_output_shape(self, input_shapes):
        """Compute output shape.

        Notes
        -----
        This is a mandatory method to implement for custom Keras layers.

        Parameters
        ----------
        input_shapes : list
            A list of the shape of inputs, which are the images at multiple resolutions and the focus x, y coordinates
            - multi_res_images : (batch_size, h, w, n_resolutions, n_channels)
            - focuses : (batch_size, 2)

        Returns
        -------
        None
            Batch size

        height : int
            Height of the sampled (output) image.

        width : int
            Width of the sampled (output) image.

        n_resolutions: int
            The number of resolution stages to use

        n_channels : int
            Number of channels.

        """
        height, width, n_resolutions, n_channels = input_shapes[0][1:]

        return None, self._tgt_h, self._tgt_w, n_resolutions, n_channels

    def call(self, tensors, mask=None):
        """Perform forward pass.

        Parameters
        ----------
        tensors : list
            A list of two elements - multi-resolutions images to we samples and the focus locaitons.
            Their shapes are the following
                - multi-resolution images : (batch_size, h, w, n_resolutions, n_channels)
                - focuses : (batch_size, 2)
        mask

        Returns
        -------
        output : K.Tensor
            Images sampled at the specified locations.
            Shape (batch_size, output_height, output_width, n_resolutions, n_channels).

        """
        imgs, focuses = tensors

        # dimensions as tensors
        batch_size = K.shape(imgs)[0]
        in_height = K.shape(imgs)[1]
        in_width = K.shape(imgs)[2]
        num_res = K.shape(imgs)[3]
        num_channels = K.shape(imgs)[4]

        sampled_imgs = self._sample_at_focus(imgs, focuses)

        return sampled_imgs

    def _sample_at_focus(self, imgs, focuses):
        raise NotImplementedError()
