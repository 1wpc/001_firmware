import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取左右图像
left_image = cv2.imread('shot_a2.jpg', cv2.IMREAD_GRAYSCALE)  # 左图
right_image = cv2.imread('shot_b2.jpg', cv2.IMREAD_GRAYSCALE)  # 右图

# 创建 StereoSGBM 对象
# minDisparity: 最小视差值
# numDisparities: 视差范围的大小，必须是16的倍数
# blockSize: 计算视差时的窗口大小（通常为奇数）
# P1 和 P2 控制代价函数的平滑度，P1 是较小的平滑参数，P2 是较大的平滑参数
# 其他参数可以根据需求调整
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 5,  # 视差范围，可以根据需要调整
    blockSize=7,
    P1=8 * 3 * 7 ** 2,  # 控制平滑度，设置较高的值会减少视差跳跃
    P2=32 * 3 * 7 ** 2,  # 控制平滑度，设置较高的值会减少视差跳跃
    disp12MaxDiff=1,  # 最大的视差差异
    preFilterCap=63,  # 预处理步骤的值
    uniquenessRatio=10,  # 唯一性比率，值越高，匹配越严格
    speckleWindowSize=100,  # 视差杂散点的窗口大小
    speckleRange=32  # 视差的范围
)

# 计算视差图
disparity = stereo.compute(left_image, right_image).astype(np.float32) / 16.0  # 除以16将计算结果恢复到正确的视差范围

# 显示视差图
plt.imshow(disparity, cmap='gray')
plt.colorbar()
plt.show()
