import cv2
import itertools, os, time
import numpy as np
from Model import get_Model
from parameter import letters
import argparse
from keras import backend as K
K.set_learning_phase(0)


def decode_label(out):   # 此函数输出的就是预测的字符串
    # out : (1,48,67）
    out_best = list(np.argmax(out[0, 2:], axis=1))  # 48个中的每一个元素都对应67个值，找出每一个元素所对应的67个值中最大的索引
    # out_best列表存储的是一个索引
    out_best = [k for k, g in itertools.groupby(out_best)]  # 移除重复的值，使48个索引值减小到8个
    outstr = ''
    for i in out_best:
        if i < len(letters):
            outstr += letters[i]
    return outstr


'''
argparse是Python中的一个常用模块，主要用于编写命令行接口
使用argparse的第一步是创建ArgumentParser对象即parser = argparse.ArgumentParser()
调用add_argument()向ArgumentParser对象添加命令行参数信息，这些信息告诉ArgumentParser对象如何处理命令行参数，之后再
调用parse.args()将返回一个对象
'''
parser = argparse.ArgumentParser()
parser.add_argument("-w", "--weight", help="weight file directory",
                    type=str, default="Final_weight.hdf5")
parser.add_argument("-t", "--test_img", help="Test image directory",   # test_img代表了测试图片的文件夹路径
                    type=str, default="./DB/test/")
args = parser.parse_args()

# Get CRNN model
model = get_Model(training=False)

try:
    model.load_weights(args.weight)
    print("...Previous weight data...")
except:
    raise Exception("No weight file!")


test_dir =args.test_img
test_imgs = os.listdir(args.test_img)
total = 0
acc = 0
letter_total = 0
letter_acc = 0
start = time.time()
for test_img in test_imgs:
    img = cv2.imread(test_dir + test_img, cv2.IMREAD_GRAYSCALE)

    img_pred = img.astype(np.float32)
    img_pred = cv2.resize(img_pred, (192, 64))    # img_pred是一个二维矩阵192列64行
    img_pred = (img_pred / 255.0) * 2.0 - 1.0
    img_pred = img_pred.T
    img_pred = np.expand_dims(img_pred, axis=-1)  # 在axis处增加一个维度
    img_pred = np.expand_dims(img_pred, axis=0)

    net_out_value = model.predict(img_pred)  # 进行检测 利用model中的predict方法

    pred_texts = decode_label(net_out_value)  # pred_texts就是最终预测的完整车牌号字符序列

    if pred_texts[0] == ' ':
        pred_texts = pred_texts[1:]
        color = '蓝牌'
        for i in range(min(len(pred_texts), len(test_img[0:-4]))):
            if pred_texts[i] == test_img[i]:
                letter_acc +=1
        letter_total += max(len(pred_texts), len(test_img[0:-4]))
        if pred_texts == test_img[0:-4]:
            acc += 1
        total += 1
        print('预测车牌颜色: %s  /  预测车牌号为: %s  / 真实车牌号为: %s' % (color, pred_texts, test_img[0:-4]))
    else:
        color = '绿牌'
        for i in range(min(len(pred_texts), len(test_img[0:-4]))):
            if pred_texts[i] == test_img[i]:
                letter_acc += 1
        letter_total += max(len(pred_texts), len(test_img[0:-4]))
        if pred_texts == test_img[0:-4]:
            acc += 1
        total += 1
        print('预测车牌颜色: %s  /  预测车牌号为: %s / 真实车牌号为: %s' % (color, pred_texts, test_img[0:-4]))


end = time.time()
total_time = (end - start)
print("平均每张车牌检测时间: ", total_time / total)
print("车牌准确率 : ", acc / total)
print("字符准确率 : ", letter_acc / letter_total)
