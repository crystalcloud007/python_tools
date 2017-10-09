# 输入给定图片，输出该图片的多数颜色值。
# 可用来提取一个图片的色板，方便根据色板进行颜色设计。
# 首先将图片转成矩阵，根据KNN算法，聚类出特征颜色信息。
# 再将颜色信息规约为指定个数的数组。
# 最后输出为N行4列的图片，就是色版图。
# 分析时，如果图片是RGBA格式，会强制转为RGB格式。
# 最终输出色彩规约后的图像和色板图像。

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# 分析用绘图函数
# def plotPixels(data, title, colors = None, N = 10000):
#     if colors is None:
#         colors = data

#     rng = np.random.RandomState(0)
#     i = rng.permutation(data.shape[0])[:N]
#     colors = colors[i]
#     R,G,B=data[i].T

#     fig,ax = plt.subplots(1,2,figsize=(16,6))
#     ax[0].scatter(R,G,color=colors,marker='.')
#     ax[0].set(xlabel='Red', ylabel='Green', xlim=(0,1), ylim=(0,1))
#     ax[1].scatter(R,B,color=colors,marker='.')
#     ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0,1), ylim=(0,1))

#     fig.suptitle(title,size=20)

print('本程序可以从图片中提取特征颜色，组成色板。')
input_filename = input('输入要分析的图片路径\n>')
panel_size = int(input('输入要总结出的颜色数量，4、8、16、32或64\n>'))

fn_arr = input_filename.split('.')
output_filename = ''.join(fn_arr[0:-1]) + '_panel.' + fn_arr[-1]
colored_filename = ''.join(fn_arr[0:-1]) + '_colored.' + fn_arr[-1]

if all([panel_size != 4, panel_size != 8, panel_size != 16, panel_size != 32, panel_size != 64]):
        print('颜色数量输入错误，强制转为4。')
        panel_size = 4


sample_img = Image.open(input_filename)
data = np.asarray(sample_img)
shape = data.shape

data = data / 255.0     # 将整个空间映射到0-1区间
data = data.reshape(shape[0] * shape[1], shape[-1])
print('完成文件预处理')

# plotPixels(data, title='Input color space')

kmeans = MiniBatchKMeans(panel_size)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

# plotPixels(data, colors=new_colors, title='Panel Space')

img_colored = new_colors.reshape(np.asarray(sample_img).shape)
img_colored_file = Image.fromarray(np.uint8((img_colored * 255.0).astype('int')))
img_colored_file.save(colored_filename)
print('完成色板统计，并输出色板化之后的图像于：' + colored_filename)

# 将像素点规约，找到色板需要的数据
color_panel_data = np.unique(new_colors, axis=0)
color_panel_data = color_panel_data * 255.0
color_panel_data = color_panel_data.astype('int')

# 将色板数据输出为图片
img_output_data = np.zeros([int(panel_size / 4) * 16, 16 * 4, 3])
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

# 绘图展示处理结果
fig, ax = plt.subplots(1,3,figsize=(16,6), subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(sample_img)
ax[0].set_title('ORIGIN', size=16)
ax[1].imshow(img_colored)
ax[1].set_title('COLORED', size=16)
ax[2].imshow(img_output_file)
ax[2].set_title('PANEL', size=16)
plt.show()