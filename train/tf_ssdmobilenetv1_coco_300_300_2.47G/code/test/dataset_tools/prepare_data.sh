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

DATA=data
SUB_DATA=${DATA}/coco2014_minival_8059

CUR_DIR=$(pwd)
echo "Entering ${DATA}..."
cd ${DATA}

echo "Prepare to download COCO train-val2014 anotation zip file..."
wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip

echo "Prepare to download COCO val2014 image zip file..."
wget -c http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
rm -f val2014.zip

echo "Entering ${CUR_DIR}"
cd ${CUR_DIR}

echo "Generating validation list..."
IDX_LIST=code/test/dataset_tools/mscoco_minival_ids.txt
DST_LIST=${SUB_DATA}/minival2014_8059.txt
python code/test/dataset_tools/gen_minival2014_list.py -idx_list ${IDX_LIST} -dst_list ${DST_LIST}

echo "Collecting validation images..."
SRC_DIR=${DATA}/val2014
DST_DIR=${SUB_DATA}/image
if [ ! -d ${DST_DIR} ]
then
    mkdir -p ${DST_DIR}
fi
while IFS= read -r filename
do
    cp ${SRC_DIR}/${filename}.jpg ${DST_DIR}/${filename}.jpg
done < ${DST_LIST}

echo "Generating validation groundtruth json..."
OLD_JSON=${DATA}/annotations/instances_val2014.json
NEW_JSON=${SUB_DATA}/minival2014_8059.json
FILTER_FILE=code/test/dataset_tools/mscoco_minival_ids.txt
python code/test/dataset_tools/gen_minival2014_json.py -old_json_file ${OLD_JSON} -new_json_file ${NEW_JSON} -filter_file ${FILTER_FILE}

echo "Finished!"
