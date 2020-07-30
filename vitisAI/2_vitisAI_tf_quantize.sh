### 
# @Author: zhangyin
 # @Email: miracleyin@live.com
 # @Date: 2020-07-28 
 # @Description: quantize frozon tf model 
 # @Dependence: tensorflow 1.15, Vitis-AI Release 1.2
 ###
#!/bin/bash


# activate DECENT_Q Python3.6 virtual environment
#conda activate decent_q3

# generate calibraion images and list file
#python generate_images.py

# remove existing files
rm -rf ./quantize_results


# run quantization
echo "#####################################"
echo "Quantize begin"
echo "Vitis AI 1.2"
echo "#####################################"

vai_q_tensorflow quantize \
  --input_frozen_graph ./trainedModels/ssd_mobilenet_innference_graph.pb \
  --input_nodes image_tensor \
  --input_shapes ?,300,300,3 \
  --output_nodes  \
  --method 1 \
  --input_fn graph_input_fn.calib_input \
  --gpu 0 \
  --calib_iter 50 \

echo "#####################################"
echo "QUANTIZATION COMPLETED"
echo "#####################################"

