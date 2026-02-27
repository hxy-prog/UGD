import cv2
import os
import numpy as np

# def images_to_video(image_folder, video_name, fps):
#     # 使用自定义排序函数确保数字按正确顺序排序
#     images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
#     images = sorted(images, key=lambda x: int(x.split('p')[0]))  # 假设文件名格式为 `number + 'r.jpg'`

#     # 从第一张图片读取尺寸信息
#     frame = cv2.imread(os.path.join(image_folder, images[0]))
#     height, width, layers = frame.shape

#     # 定义视频编码和创建VideoWriter对象
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'XVID'
#     video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

#     # 读取每一张图片，添加到视频中
#     for image in images:
#         video.write(cv2.imread(os.path.join(image_folder, image)))

#     # 释放VideoWriter对象
#     video.release()

# # 使用函数
# image_folder = '/home/hexingyang/DDP/segmentation/ddimvis'  # 图片文件夹路径
# video_name = 'denoise.mp4'  # 输出视频文件名
# fps = 30  # 帧率，可以根据需要调整
# images_to_video(image_folder, video_name, fps)
mask=cv2.imread('41.png',cv2.IMREAD_GRAYSCALE)
mask=255-mask
cv2.imwrite('true.png',mask)