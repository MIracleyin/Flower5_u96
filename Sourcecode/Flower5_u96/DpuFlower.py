import PIL
import IPython
from io import BytesIO as StringIO
from IPython.display import display
from IPython.display import clear_output
import cv2
from dnndk import n2cube
import numpy as np
from numpy import float32
import os
import matplotlib.pyplot as plt
import time

class DpuFlower(object):
    # TODO:dpu_input_node="input", dpu_output_node="output", dpu_img_size=128 需要调整
    def __init__(self, dpu_task, dpu_input_node="conv2d_Conv2D", dpu_output_node="y_out_MatMul", dpu_img_size=128):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 160)
        self.cap.set(4, 120)
        print(self.cap.get(3), self.cap.get(4))
        print(self.cap.get(cv2.CAP_PROP_FPS))

        self.dpuInputNode = dpu_input_node
        self.dpuOutputNode = dpu_output_node
        self.dpuTaks = dpu_task
        self.dpuImgSize = dpu_img_size

    def get_image(self, idx=0):
        """
        get a image from sensor, donot care which kind of cam is used.
        Args:
            idx: the index of sensor, default is 0
        """ 
        if idx == 0:
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                print("Please connect the camera!")
                return False
        else:
            print("The index should be 0!")
    def dpuFlowerPredictSoftmax(self, img_input):
        img_scale = cv2.resize(img_input, (self.dpuImgSize, self.dpuImgSize), interpolation=cv2.INTER_CUBIC)
        img1_scale = np.array(img_scale, dtype='float32')
        if np.max(img1_scale) > 1:
            img1_scale = img1_scale / 255.
        input_len = img1_scale.shape[0] * img1_scale.shape[1] * img1_scale.shape[2] #输入数据的长度
        #input_len = n2cube.dpuGetInputTensorSize(task, KERNEL_CONV_INPUT)
        # Set DPU Task input Tensor with data from a CPU memory block.
        # 设定 DPU 任务输入张量以及所需的 CPU 内存
        n2cube.dpuSetInputTensorInHWCFP32(self.dpuTaks, self.dpuInputNode, img1_scale, input_len)
        # Launch the running of DPU Task.
        n2cube.dpuRunTask(self.dpuTaks)
        # 以下代码有一些问题
        # softmax需要4个参数
        # 后期可以移到dpuFlowerSetSoftmax
        conf = n2cube.dpuGetOutputTensorAddress(self.dpuTaks, self.dpuOutputNode) 
        channel = n2cube.dpuGetOutputTensorChannel(self.dpuTaks, self.dpuOutputNode)
        outScale = n2cube.dpuGetOutputTensorScale(self.dpuTaks, self.dpuOutputNode)
        size = n2cube.dpuGetOutputTensorSize(self.dpuTaks, self.dpuOutputNode)
        ################
        softmax = n2cube.dpuRunSoftmax(conf, channel, size//channel, outScale)
        pdt = np.argmax(softmax, axis=0)
        return pdt

class CommonFunction(object):
    @classmethod
    def img2display(cls, img_mat):
        ret, png = cv2.imencode('.png', img_mat)
        encoded = base64.b64encoded.decode('ascii')
        return Image(data=encoded.decode("ascii"))

    @classmethod
    def show_img_jupyter(cls, img_mat):
        img_mat = cv2.cvtColor(img_mat, cv2.COLOR_BGR2RGB)
        f = StringIO()
        PIL.Image.fromarray(img_mat).save(f, 'png')
        IPython.display.display(IPython.display.Image(data=f.getvalue()))
    
    @classmethod
    def clear_output(cls):
        clear_output(wait=True)
    








