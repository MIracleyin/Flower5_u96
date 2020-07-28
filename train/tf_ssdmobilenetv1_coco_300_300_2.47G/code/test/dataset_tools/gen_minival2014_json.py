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

import os
import json
import argparse


def gen_sample_gt_json(old_json_file, new_json_file, filter_file):

    filter_list = []
    with open(filter_file, "r") as f_fil:
        lines = f_fil.readlines()
        for line in lines:
            filter_list.append(line.strip())
    
    with open(old_json_file, "r") as f_old:
        old_json_str = f_old.read()
    parse_object = json.loads(old_json_str)
    
    # print(parse_object.keys())
    new_info = parse_object['info']
    new_license = parse_object['licenses']
    new_categories = parse_object['categories']
    new_annotations = parse_object['annotations']
    new_images = parse_object['images']
    
    index = []
    # print(type(filter_list[0]))
    # print(len(parse_object['images']))
    for i in range(len(parse_object['images'])):
        if str(parse_object['images'][i]['id']) in filter_list:
            index.append(i)
    
    new_data = {}
    new_images = [new_images[ind] for ind in index]
    
    index = []
    # print(len(parse_object['annotations']))
    # print(parse_object['annotations'][0])
    for i in range(len(parse_object['annotations'])):
        if str(parse_object['annotations'][i]['image_id']) in filter_list:
            index.append(i)
    new_annotations = [new_annotations[ind] for ind in index]
    
    new_data['info'] = new_info
    new_data['licenses'] = new_license
    new_data['categories'] = new_categories
    new_data['annotations'] = new_annotations
    new_data['images'] = new_images
    with open(new_json_file, 'w') as f_det:
        f_det.write(json.dumps(new_data))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script to generate the validation groundtruth json')
    parser.add_argument('-old_json_file', default='data/annotations/instances_val2014.json')
    parser.add_argument('-new_json_file', default='data/coco2014_minival_8059/minival2014_8059.json')
    parser.add_argument('-filter_file', default='code/test/dataset_tools/mscoco_minival_ids.txt')
    args = parser.parse_args()

    dst_dir, dst_file = os.path.split(args.new_json_file)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    gen_sample_gt_json(args.old_json_file, args.new_json_file, args.filter_file)
    print("Validation groundtruth is saved as {}".format(args.new_json_file))
