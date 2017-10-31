#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/10/31
  
"""
import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

item = [
    '希特勒回来了', '死侍', '房间', '龙虾', '大空头',
    '极盗者', '裁缝', '八恶人', '实习生', '间谍之桥',
]

user = ['五柳君', '帕格尼六', '木村静香', 'WTF', 'airyyouth',
        '橙子c', '秋月白', 'clavin_kong', 'olit', 'You_某人',
        '凛冬将至', 'Rusty', '噢！你看！', 'Aron', 'ErDong Chen'
        ]

RATE_MATRIX = np.array(
        [[5, 5, 3, 0, 5, 5, 4, 3, 2, 1, 4, 1, 3, 4, 5],
         [5, 0, 4, 0, 4, 4, 3, 2, 1, 2, 4, 4, 3, 4, 0],
         [0, 3, 0, 5, 4, 5, 0, 4, 4, 5, 3, 0, 0, 0, 0],
         [5, 4, 3, 3, 5, 5, 0, 1, 1, 3, 4, 5, 0, 2, 4],
         [5, 4, 3, 3, 5, 5, 3, 3, 3, 4, 5, 0, 5, 2, 4],
         [5, 4, 2, 2, 0, 5, 3, 3, 3, 4, 4, 4, 5, 2, 5],
         [5, 4, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0],
         [5, 4, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
         [5, 4, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
         [5, 4, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]
)

nmf_model = NMF(n_components=2)  # 有两个主题
item_dis = nmf_model.fit_transform(RATE_MATRIX)
user_dis = nmf_model.components_

# print(item_dis)
print('用户的主题分布：')
print(user_dis)
print('电影的主题分布：')
print(item_dis)

# plt1 = plt
# plt1.plot(item_dis[:, 0], item_dis[:, 1], 'ro')
# plt1.draw()  # 直接画出矩阵，只打了点，下面对图plt1进行一些设置
#
# plt1.xlim((-1, 3))
# plt1.ylim((-1, 3))
# plt1.title(u'the distribution of items (NMF)')  # 设置图的标题
#
# count = 1
# zipitem = zip(item, item_dis)  # 把电影标题和电影的坐标联系在一起
#
# for item in zipitem:
#     item_name = item[0]
#     data = item[1]
#     plt1.text(data[0], data[1], item_name,
#               horizontalalignment='center',
#               verticalalignment='top')
# plt.show()
#
# user_dis = user_dis.T  # 把转置用户分布矩阵
# plt1 = plt
# plt1.plot(user_dis[:, 0], user_dis[:, 1], 'ro')
# plt1.xlim((-1, 3))
# plt1.ylim((-1, 3))
# plt1.title(u'the distribution of user (NMF)')  # 设置图的标题
#
# zipuser = zip(user, user_dis)  # 把电影标题和电影的坐标联系在一起
# for user in zipuser:
#     user_name = user[0]
#     data = user[1]
#     plt1.text(data[0], data[1], user_name,
#               horizontalalignment='center',
#               verticalalignment='top')
#
# plt1.show()  # 直接画出矩阵，只打了点，下面对图plt1进行一些设置

rec_mat = np.dot(item_dis, user_dis)
filter_matrix = RATE_MATRIX < 1e-8
print('重建矩阵，并过滤掉已经评分的物品：')
rec_filter_mat = (filter_matrix * rec_mat).T
print(rec_filter_mat)