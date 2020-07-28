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

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf

def combined_static_and_dynamic_shape(tensor):
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape

def get_center_coordinates_and_sizes(box_corners, scope=None):
  """Computes the center coordinates, height and width of the boxes.

  Args:
    scope: name scope of the function.

  Returns:
    a list of 4 1-D tensors [ycenter, xcenter, height, width].
  """
  with tf.name_scope(scope, 'get_center_coordinates_and_sizes'):
    ymin, xmin, ymax, xmax = tf.unstack(tf.transpose(box_corners))
    width = xmax - xmin
    height = ymax - ymin
    ycenter = ymin + height / 2.
    xcenter = xmin + width / 2.
    return [ycenter, xcenter, height, width]

def faster_rcnn_box_coder_decode(rel_codes, anchors, scale_factors):
  """Decode relative codes to boxes.

  Args:
    rel_codes: a tensor representing N anchor-encoded boxes.
    anchors: BoxList of anchors.

  Returns:
    boxes: BoxList holding N bounding boxes.
  """
  ycenter_a, xcenter_a, ha, wa = get_center_coordinates_and_sizes(anchors)

  ty, tx, th, tw = tf.unstack(tf.transpose(rel_codes))
  if scale_factors:
    ty /= scale_factors[0]
    tx /= scale_factors[1]
    th /= scale_factors[2]
    tw /= scale_factors[3]
  w = tf.exp(tw) * wa
  h = tf.exp(th) * ha
  ycenter = ty * ha + ycenter_a
  xcenter = tx * wa + xcenter_a
  ymin = ycenter - h / 2.
  xmin = xcenter - w / 2.
  ymax = ycenter + h / 2.
  xmax = xcenter + w / 2.
  return tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))


def batch_decode(box_encodings, anchors, scale_factors):
  """Decodes a batch of box encodings with respect to the anchors.
  """
  combined_shape = combined_static_and_dynamic_shape(box_encodings)
  batch_size = combined_shape[0]
  tiled_anchor_boxes = tf.tile(tf.expand_dims(anchors, 0), [batch_size, 1, 1])
  tiled_anchors_boxlist = tf.reshape(tiled_anchor_boxes, [-1, 4])
  decoded_boxes = faster_rcnn_box_coder_decode(tf.reshape(box_encodings, [-1, 4]),
                        tiled_anchors_boxlist, scale_factors)
  decoded_boxes = tf.reshape(decoded_boxes, tf.stack([combined_shape[0], combined_shape[1], 4]))
  return decoded_boxes

