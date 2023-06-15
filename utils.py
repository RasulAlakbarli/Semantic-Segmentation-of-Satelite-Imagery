from typing import Any, Tuple
import os
import cv2
import rasterio
import geopandas as gpd
import numpy as np
import tensorflow as tf
from skimage.color import gray2rgb
from rasterio import mask as Mask

class Utils:
    @staticmethod
    def mask_raster(raster_file_path: str, shape_file_path: str) -> Tuple[Any, Any]:
        """
        Generate binary mask from raster and array of geometry objects.
        
        Returns: Tuple[np.array: binary mask array, aff: affine.Affine] 
        """
        
        raster = rasterio.open(raster_file_path, mode='r')
        vectorfile = gpd.read_file(shape_file_path, mode='r')
        geometry_objects = np.array(vectorfile["geometry"].values)
        
        return Mask.raster_geometry_mask(raster, geometry_objects, invert=True)


    @staticmethod
    def load_image_w_mask(image_dir, mask_dir):
        '''
        Reads image and creates corresponding mask.
        Args: Image file path, Mask file path
        Returns: np.ndarray: image, np.ndarray: binary mask
        '''
        # Load image
        image = cv2.imread(image_dir)
        
        # Load mask
        binary_mask, _, _ = Utils.mask_raster(image_dir, mask_dir)
        binary_mask = gray2rgb(binary_mask)
        binary_mask = tf.cast(binary_mask, tf.uint8)
        
        return image, binary_mask

    @staticmethod
    def patcher(image, binary_mask):
        '''
        Patchifies the image and its mask
        Args: np.ndarray: image, np.ndarray: binary mask
        Returns: tensorflow.python.framework.ops.EagerTensor: image_patches, tensorflow.python.framework.ops.EagerTensor: mask_patches
        '''
        image_patches = tf.image.extract_patches(images=tf.expand_dims(image, 0),
                                        sizes=[1, 128, 128, 1],
                                        strides=[1, 128, 128, 1],
                                        rates=[1, 1, 1, 1],
                                        padding='SAME')


        mask_patches = tf.image.extract_patches(images=tf.expand_dims(binary_mask, axis=0),
                                        sizes=[1, 128, 128, 1],
                                        strides=[1, 128, 128, 1],
                                        rates=[1, 1, 1, 1],
                                        padding='SAME')

        return image_patches, mask_patches

    @staticmethod
    def save_tiled(raster_file_paths, shape_file_paths):
        '''
        Patchifies and saves both images and masks
        
        Args: Image file path, Mask file path
        Returns: Saved images in specified directory
        '''
        image_tile_dir = "128x_tiles/train_images/train/"
        mask_tile_dir = "128x_tiles/train_masks/train/"
        if not os.path.exists(image_tile_dir):
            os.makedirs(image_tile_dir)
        if not os.path.exists(mask_tile_dir):
            os.makedirs(mask_tile_dir)
            
        for img, msk in zip(raster_file_paths, shape_file_paths):
            image, binary_mask = Utils.load_image_w_mask(img, msk) 
            patches = Utils.patcher(image, binary_mask) 
            patched_image = patches[0]
            patched_mask = patches[1]
            patched_images = tf.reshape(patched_image, shape=(patched_image.shape[1]*patched_image.shape[2], 128, 128, 3)).numpy().astype("uint8")
            patched_masks = tf.reshape(patched_mask, shape=(patched_mask.shape[1]*patched_mask.shape[2], 128, 128, 3)).numpy().astype("uint8")     
            for i, (img_tile, msk_tile) in enumerate(zip(patched_images, patched_masks)):
                cv2.imwrite(image_tile_dir+"tile-{}-{}-{}.png".format(i, patched_image.shape[1], patched_image.shape[2]), img_tile)
                cv2.imwrite(mask_tile_dir+"tile-{}-{}-{}.png".format(i, patched_image.shape[1], patched_image.shape[2]), msk_tile)