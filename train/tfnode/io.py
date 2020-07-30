import tensorflow as tf
import os

model_name = 'frozen_inference_graph_mobilenet.pb'
 
def create_graph():
    with tf.io.gfile.GFile(os.path.join(model_name), 'rb') as f:
        graph_def =  tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
 
create_graph()
tensor_name_list = [tensor.name for tensor in tf.compat.v1.get_default_graph().as_graph_def().node]
filename = model_name[0:-2]
with open(filename,'w') as f:
    for tensor_name in tensor_name_list:
        f.write(tensor_name + '\n')

