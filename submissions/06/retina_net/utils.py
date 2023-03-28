import numpy as np
import tensorflow as tf

from typing import Tuple, Dict


def prepare_data(example: Dict[tf.Tensor]):
    """Do a single image sample preprocessing re-sizing, augmentation, bounding boxes, label encoding, etc.)

    Args:
        example (Dict[tf.Tensor]): 
            Image sample dictionary consisting of image data ("image") and ground truth - bounding boxes("bboxes") and classes("labels").

    Return:
        processed_sample (Tuple[tf.Tensor]): Preprocessed image sample as a tuple of - processed_image, true_cls, true_boxes
    """         
    processed_sample = None
    true_boxes = None
    true_cls = None
    raise NotImplementedError

    return processed_sample, true_cls, true_boxes



def encode_image_batch(image_batch: tf.Tensor, cls_batch: tf.Tensor, boxes_batch: tf.Tensor):
    """_summary_

    Args:
        image_batch (tf.Tensor): _description_
        cls_batch (tf.Tensor): _description_
        boxes_batch (tf.Tensor): _description_

    Raises:
        NotImplementedError: _description_
    """    
    
    raise NotImplementedError

