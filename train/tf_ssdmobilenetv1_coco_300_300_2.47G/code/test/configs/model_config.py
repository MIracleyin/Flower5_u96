# (c) Copyright 2019 Xilinx, Inc. All rights reserved.
#
# This file contains confidential and proprietary information
# of Xilinx, Inc. and is protected under U.S. and
# international copyright and other intellectual property
# laws.
#
# DISCLAIMER
# This disclaimer is not a license and does not grant any
# rights to the materials distributed herewith. Except as
# otherwise provided in a valid license issued to you by
# Xilinx, and to the maximum extent permitted by applicable
# law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
# WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
# AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
# BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
# INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
# (2) Xilinx shall not be liable (whether in contract or tort,
# including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature
# related to, arising under or in connection with these
# materials, including for any direct, or any indirect,
# special, incidental, or consequential loss or damage
# (including loss of data, profits, goodwill, or any type of
# loss or damage suffered as a result of any action brought
# by a third party) even if such damage or loss was
# reasonably foreseeable or Xilinx had been advised of the
# possibility of the same.
#
# CRITICAL APPLICATIONS
# Xilinx products are not designed or intended to be fail-
# safe, or for use in any application requiring fail-safe
# performance, such as life-support or safety devices or
# systems, Class III medical devices, nuclear facilities,
# applications related to the deployment of airbags, or any
# other applications that could lead to death, personal
# injury, or severe property or environmental damage
# (individually and collectively, "Critical
# Applications"). Customer assumes the sole risk and
# liability of any use of Xilinx products in Critical
# Applications, subject only to applicable laws and
# regulations governing limitations on product liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
# PART OF THIS FILE AT ALL TIMES.

from easydict import EasyDict
import tensorflow as tf

class SSDMobilenetV1Config(object):
  def __init__(self, weights_path):
    self.pb_file = weights_path
    self.height = 300 
    self.width = 300 
    self.input_tensor = "image_tensor:0"
    self.box_encoding_tensor = "concat:0"
    self.class_score_tensor = "concat_1:0"

    self.feature_extractor_type = "ssd_mobilenet_v1"

    self.anchor_type = "ssd_anchor_generator"
    self.anchor_config = EasyDict()
    self.anchor_config.num_layers = 6 
    self.anchor_config.min_scale = 0.2 
    self.anchor_config.max_scale = 0.95
    self.anchor_config.scales = []
    self.anchor_config.aspect_ratios = [1.0, 2.0, 0.5, 3.0, 0.3333]
    self.anchor_config.interpolated_scale_aspect_ratio=1.0
    self.anchor_config.base_anchor_size=[1.0, 1.0]
    self.anchor_config.anchor_strides=None
    self.anchor_config.anchor_offsets=None
    self.anchor_config.reduce_boxes_in_lowest_layer=True

    self.scale_factors = [10.0, 10.0, 5.0, 5.0]
    self.feature_map_spatial_dims = [(19, 19), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)] 

    self.nms_config = EasyDict()
    self.nms_config.score_threshold = 0.005
    self.nms_config.iou_threshold = 0.6 
    self.nms_config.max_detections_per_class = 100 
    self.nms_config.max_total_detections = 100 

    self.score_fn = tf.sigmoid
    self.logit_scale = 1.0 
