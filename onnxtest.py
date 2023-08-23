import onnxruntime as ort
import cv2
import numpy as np
# 读取图片
mean=[123.675, 116.28, 103.53]
std=[58.395, 57.12, 57.375]
img_path = '/media/meiguiz/HIKSEMI/mmsegmentation/data/trosd/test_input/cg_real_test_d415_000000012_3_v.png'
input_shape = (640,480)
img = cv2.imread(img_path)
print(img.shape)
img = cv2.resize(img, input_shape)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = (img - mean) / std
img = img.astype(np.float32)

# 注意输入的图像的type一定要是 numpy 而不是 tensor。此处为np.float32，因为torch模型是float32，要保持一致。
img = img.transpose(2,0,1).reshape(1, 3, 480, 640)  # HWC->BCHW

# 加载 onnx
onnx_path= '/media/meiguiz/HIKSEMI/mmsegmentation/tools/tmp.onnx'
sess = ort.InferenceSession(onnx_path,providers=['CUDAExecutionProvider']) # 'CPUExecutionProvider'
input_name = sess.get_inputs()[0].name
output_name = [output.name for output in sess.get_outputs()]

# 推理
outputs = sess.run(output_name, {input_name:img})
print(outputs[0][0].max())
imgout = np.zeros((480, 640), np.uint8)
print(imgout.shape[0],imgout.shape[1])
for i in range(0,imgout.shape[0]-1):
 for j in range(0,imgout.shape[1]-1):
  imgout[i,j]=outputs[0][0][0][i][j]*50
cv2.imshow("image out", imgout)
cv2.waitKey(0)

