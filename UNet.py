import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, concatenate, BatchNormalization, Activation
from tensorflow.keras.callbacks import ModelCheckpoint

class UNet(Model):
    """
    Implementation of the U-Net model for image segmentation.

    The class provides methods to create the input tensor, apply convolutional and deconvolutional layers,
    perform pooling and merging operations, and define the U-Net model architecture.


    Args:
    - img_height (int): Height of the input tensor.
    - img_width (int): Width of the input tensor.
    
    Returns:
    - U-Net model
    
    Usage:
        unet = UNet(img_height=256, img_width=256)
        model = unet.model()
        
    """
    def __init__(self, img_height, img_width):
        super(UNet, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        
    def conv_block(self, tensor, nfilters, size=3, padding='same', initializer="he_normal"):
        x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x


    def deconv_block(self, tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
        y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
        y = concatenate([y, residual], axis=3)
        y = self.conv_block(y, nfilters)
        return y


    def model(self, nclasses=2, filters=32):
        # down
        input_layer = Input(shape=(self.img_height, self.img_width, 3), name='image_input')
        conv1 = self.conv_block(input_layer, nfilters=filters)
        conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = self.conv_block(conv1_out, nfilters=filters*2)
        conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = self.conv_block(conv2_out, nfilters=filters*4)
        conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = self.conv_block(conv3_out, nfilters=filters*8)
        conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
        conv4_out = Dropout(0.5)(conv4_out)
        conv5 = self.conv_block(conv4_out, nfilters=filters*16)
        conv5 = Dropout(0.5)(conv5)
        # up
        deconv6 = self.deconv_block(conv5, residual=conv4, nfilters=filters*8)
        deconv6 = Dropout(0.5)(deconv6)
        deconv7 = self.deconv_block(deconv6, residual=conv3, nfilters=filters*4)
        deconv7 = Dropout(0.5)(deconv7) 
        deconv8 = self.deconv_block(deconv7, residual=conv2, nfilters=filters*2)
        deconv9 = self.deconv_block(deconv8, residual=conv1, nfilters=filters)
        # output
        output_layer = Conv2D(filters=nclasses, kernel_size=(1, 1))(deconv9)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Activation('softmax')(output_layer)

        model = Model(inputs=input_layer, outputs=output_layer, name='Unet')
        return model