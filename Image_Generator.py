import cv2
import os, random
import numpy as np
from parameter import letters


# # Input data generator
def labels_to_text(labels):     # 字母的索引 -> text (string)
    return ''.join(list(map(lambda x: letters[int(x)], labels)))


def text_to_labels(text):      # 将text转换为letters数列中的索引值

    if len(text) == 7:
        text = ' ' + text
        indexing = list(map(lambda x: letters.index(x), text))
    else:
        indexing = list(map(lambda x: letters.index(x), text))

    return indexing  # str1.index(str2) index方法检测字符串str1中是否包含子字符串 str2。
    # 是为了将text字符串中的字符加以验证看这些字符是否在letters列表中，并返回按按顺序的text中的字符在letters中所在位置（索引）的一个列表。


class TextImageGenerator:
    def __init__(self, img_dirpath, img_w, img_h,
                 batch_size, downsample_factor, max_text_len=8):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        self.img_dirpath = img_dirpath                  # 图像文件夹路径
        self.img_dir = os.listdir(self.img_dirpath)     # 返回图像列表
        self.n = len(self.img_dir)                      # 图片的数目
        self.indexes = list(range(self.n))              # 返回一个从0到n-1的列表
        self.cur_index = 0
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))  # zeros是返回来一个给定形状和类型的用0填充的数组
        # 这里是返回一个三维的零数组，三个维度分别是图片数量，图片像素宽，图片像素高
        self.texts = []

    # 从opencv中的样本中读取并保存图像列表，将标签保存在文本中
    def build_data(self):
        print(self.n, " Image Loading start...")
        for i, img_file in enumerate(self.img_dir):
            # enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据所在列表中的排名和数据名称，一般用在 for 循环当中。
            # 由于enumerate同时列出两个参数，所以for循环中有i和img_file两个参数
            img = cv2.imread(self.img_dirpath + img_file, cv2.IMREAD_GRAYSCALE)
            # imread是加载返回一张图片
            # cv2.IMREAD_GRAYSCALE是加载一张灰度图
            img = cv2.resize(img, (self.img_w, self.img_h))  # 将图片缩放为192*64的全新大小的图片，这里resize返回的是一个二维向量
            img = img.astype(np.float32)  # 将img转换为浮点数格式
            img = (img / 255.0) * 2.0 - 1.0  # 归一化

            self.imgs[i, :, :] = img  # 将img图像信息保存到self.imgs这个三维数组里面
            self.texts.append(img_file[0:-4])  # self.texts存储的是由每个车牌名字前七个车牌字符组成的元素的列表，最终长度为n
        print(len(self.texts) == self.n)
        print(self.n, " Image Loading finish...")

    def next_sample(self):      # index max -> 把它设为0
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)  # shuffle() 方法将序列的所有元素随机排序。且shuffle必须使用random调用
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]
        # 返回目前的一张图片和该图片的标签名

    def next_batch(self):       # 导入尽可能多的batch size
        while True:
            X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])     # (bs, 192, 64, 1)
            # np.ones的用法与np.zeros的用法一模一样
            Y_data = np.ones([self.batch_size, self.max_text_len])             # (bs, 7)
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)  # (bs, 1)
            # //是除法下取整的意思，如5//2=2
            label_length = np.zeros((self.batch_size, 1))           # (bs, 1)

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T   # 对矩阵转置
                img = np.expand_dims(img, -1)  # 增加一个维度
                X_data[i] = img
                Y_data[i] = text_to_labels(text)  # Y_data成为了一个batch_size行,8(车牌号字符个数)列的一个矩阵
                label_length[i] = 8

            # 复制到字典格式
            inputs = {
                'the_input': X_data,  # (bs, 192, 64, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length,  # (bs, 1)
                'label_length': label_length  # (bs, 1)
            }
            outputs = {'ctc': np.zeros([self.batch_size])}   # (bs, 1) -> 所有元素  0
            yield (inputs, outputs)
