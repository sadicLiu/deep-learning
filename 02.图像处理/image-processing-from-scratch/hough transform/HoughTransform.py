# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


def lines_detector_hough(edge, theta_dim=None, dist_step=None, threshold=None, halfThetaWindowSize=2,
                         halfDistWindowSize=None):
    """
    :param edge: 经过边缘检测得到的二值图
    :param theta_dim: hough空间中theta轴的刻度数量(将[0,pi)均分为多少份),反应theta轴的粒度,越大粒度越细
    :param dist_step: hough空间中dist轴的划分粒度,即dist轴的最小单位长度
    :param threshold: 投票表决认定存在直线的起始阈值
    :return: 返回检测出的所有直线的参数(theta,dist)
    @author: bilibili-会飞的吴克
    """
    img_size = edge.shape
    if theta_dim is None:
        theta_dim = 90
    if dist_step is None:
        dist_step = 1
    max_dist = np.sqrt(img_size[0] ** 2 + img_size[1] ** 2)  # 对角线的长度
    dist_dim = int(np.ceil(max_dist / dist_step))

    if halfDistWindowSize is None:
        halfDistWindowSize = int(dist_dim / 50)

    # theta的范围是[0,pi). 在这里将[0,pi)进行了线性映射.类似的,也对Dist轴进行了线性映射
    accumulator = np.zeros((theta_dim, dist_dim))
    print(accumulator.shape)

    sin_theta = [np.sin(t * np.pi / theta_dim) for t in range(theta_dim)]
    cos_theta = [np.cos(t * np.pi / theta_dim) for t in range(theta_dim)]

    for i in range(img_size[0]):
        for j in range(img_size[1]):
            # 传入的图片是二值化之后的图片, 如果当前像素不是黑色的, 才进行霍夫变换
            if not edge[i, j] == 0:
                for k in range(theta_dim):
                    index_dist = int(round((i * cos_theta[k] + j * sin_theta[k]) / max_dist * dist_dim))
                    accumulator[k][index_dist] += 1

    # 阈值化
    M = accumulator.max()
    if threshold is None:
        threshold = int(M * 2.3875 / 10)
    # result中存放投票表中大于阈值的位置的索引值, dim0=2, 代表xs和ys, dim1是大于阈值的数量
    result = np.array(np.where(accumulator > threshold))

    # nms
    temp = [[], []]
    for i in range(result.shape[1]):
        # 虽然叫八邻域, 但是这里的大小是根据halfDistWindowSize和halfThetaWindowSize调整的
        eight_neiborhood = accumulator[
                           max(0, result[0, i] - halfThetaWindowSize + 1):
                           min(result[0, i] + halfThetaWindowSize, theta_dim),
                           max(0, result[1, i] - halfDistWindowSize + 1):
                           min(result[1, i] + halfDistWindowSize, dist_dim)]
        # 如果当前位置的票数大于邻域内的所有票数, 则保留当前位置(也就是保留该位置代表的直线)
        if (accumulator[result[0, i], result[1, i]] >= eight_neiborhood).all():
            temp[0].append(result[0, i])
            temp[1].append(result[1, i])
    result = np.array(temp)

    result = result.astype(np.float64)
    result[0] = result[0] * np.pi / theta_dim
    result[1] = result[1] * max_dist / dist_dim

    return result


def draw_lines(lines, edge, color=(255, 0, 0), err=3):
    if len(edge.shape) == 2:
        result = np.dstack((edge, edge, edge))
    else:
        result = edge
    Cos = np.cos(lines[0])
    Sin = np.sin(lines[0])

    for i in range(edge.shape[0]):
        for j in range(edge.shape[1]):
            e = np.abs(lines[1] - i * Cos - j * Sin)
            if (e < err).any():
                result[i, j] = color

    return result


if __name__ == '__main__':
    pic_path = './HoughImg/'
    pics = os.listdir(pic_path)

    for i in pics:
        if i.endswith(".jpeg") or i.endswith(".jpg"):
            # img = plt.imread(pic_path + i)
            img = cv2.imread(pic_path + i)
            # cv2.imshow("origin image", img)
            # cv2.waitKey()

            blurred = cv2.GaussianBlur(img, (3, 3), 0)
            # cv2.imshow("blurred image", blurred)
            # cv2.waitKey()

            if not len(blurred.shape) == 2:
                gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
            else:
                gray = blurred
            edge = cv2.Canny(gray, 50, 150)  # 二值图 (0 或 255) 得到 canny边缘检测的结果
            # cv2.imshow("edge", edge)
            # cv2.waitKey()

            lines = lines_detector_hough(edge)
            final_img = draw_lines(lines, blurred)
            cv2.imshow("final image", final_img)
            cv2.waitKey()

        break
