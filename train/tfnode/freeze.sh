# From TensorFlow souce: tensorflow/tensorflow/python/tools
python3 freeze_graph.py --input_graph=ssd_mobilenet_innference_graph.pb \
	                    --input_checkpoint=model.ckpt-8361242 \
	                    --output_graph=/tmp/frozen_graph.pb \
    	                --output_node_names=softmax