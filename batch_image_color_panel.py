# 输入给定图片，输出该图片的多数颜色值。
# 可用来提取一个图片的色板，方便根据色板进行颜色设计。
# 首先将图片转成矩阵，根据KNN算法，聚类出特征颜色信息。
# 再将颜色信息规约为指定个数的数组。
# 最后输出为N行4列的图片，就是色版图。
# 分析时，如果图片是RGBA格式，会强制转为RGB格式。
# 最终输出色彩规约后的图像和色板图像。
# 将文件夹内的所有图片都提取一遍
# ZHR 20190927

from PIL import Image
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import os

def extractColorPanel(input_filename, kmeans):
    print('开始处理文件：' + input_filename)
    fn_arr = input_filename.split('.')
    if all([fn_arr[-1].lower()!='jpg', fn_arr[-1].lower()!='jpeg', fn_arr[-1].lower()!='png', fn_arr[-1].lower()!='bmp']):
        print('该文件不是图像文件')
        return

    output_filename = ''.join(fn_arr[0:-1]) + '_panel.' + fn_arr[-1]
    colored_filename = ''.join(fn_arr[0:-1]) + '_colored.' + fn_arr[-1]
    sample_img = Image.open(input_filename)
    data = np.asarray(sample_img)
    shape = data.shape

    data = data / 255.0     # 将整个空间映射到0-1区间
    data = data.reshape(shape[0] * shape[1], shape[-1])
    print('完成文件预处理')

    kmeans.fit(data)
    new_colors = kmeans.cluster_centers_[kmeans.predict(data)]
    img_colored = new_colors.reshape(np.asarray(sample_img).shape)
    img_colored_file = Image.fromarray(np.uint8((img_colored * 255.0).astype('int')))
    img_colored_file.save(colored_filename)
    print('完成色板统计，并输出色板化之后的图像于：' + colored_filename)    

    # 将像素点规约，找到色板需要的数据
    color_panel_data = np.unique(new_colors, axis=0)
    color_panel_data = color_panel_data * 255.0
    color_panel_data = color_panel_data.astype('int')

    # 将色板数据输出为图片
    img_output_data = np.zeros([int(16 / 4) * 16, 16 * 4, 3])
    index_row = 0
    index_col = 0

    # 整理输出阵列
    for i in range(color_panel_data.shape[0]):
        if i > 0 and i % 4 == 0:
            index_row += 16
            index_col = 0

        img_output_data[index_row:index_row+16, index_col:index_col+16] = color_panel_data[i,:3]
        index_col += 16

    # 输出阵列转图片，并存储至文件
    img_output_file = Image.fromarray(np.uint8(img_output_data))
    img_output_file.save(output_filename)
    print('已保存文件至：' + output_filename)

print('---------------------------------')
print('#\t欢迎使用图像批量提取色板程序')
print('#\t仅支持jpg、jpeg、png和bmp文件格式')
print('---------------------------------\n')

filepath = input('请输入图像文件夹完整路径，以分隔符“/”作为结尾。\n>')
filenames = os.listdir(filepath)
kmeans = MiniBatchKMeans(16)

task_count = len(filenames)
print('文件路径：' + filepath + '\n共' + str(task_count) + '个任务。\n----------\n')

for i in range(task_count):
    extractColorPanel(filepath + filenames[i], kmeans)
    print('完成进度： ' + str(i+1) + ' / ' + str(task_count) + '\n----------\n')

print('任务完成。')