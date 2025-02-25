import os

classesIdKey = {
    0: '二尾蛱蝶',
    1: '光肩星天牛',
    2: '双条杉天牛',
    3: '大田鳖',
    4: '大青叶蝉',
    5: '斑衣蜡蝉',
    6: '松墨天牛',
    7: '柳蓝叶甲',
    8: '桃红颈天牛',
    9: '桃蛀螟',
    10: '桑天牛',
    11: '玉带斑蛾',
    12: '白星花金龟',
    13: '竹节虫',
    14: '红天蛾',
    15: '红缘灯蛾',
    16: '绿尾大蚕蛾',
    17: '茶翅蝽',
    18: '草履蚧',
    19: '菜蝽',
    20: '虎斑蝶',
    21: '赤条蝽',
    22: '马尾松毛虫',
    23: '麻皮蝽',
    24: '黑蚱蝉',
}

classesNameKey = {
    '二尾蛱蝶': 0,
    '光肩星天牛': 1,
    '双条杉天牛': 2,
    '大田鳖': 3,
    '大青叶蝉': 4,
    '斑衣蜡蝉': 5,
    '松墨天牛': 6,
    '柳蓝叶甲': 7,
    '桃红颈天牛': 8,
    '桃蛀螟': 9,
    '桑天牛': 10,
    '玉带斑蛾': 11,
    '白星花金龟': 12,
    '竹节虫': 13,
    '红天蛾': 14,
    '红缘灯蛾': 15,
    '绿尾大蚕蛾': 16,
    '茶翅蝽': 17,
    '草履蚧': 18,
    '菜蝽': 19,
    '虎斑蝶': 20,
    '赤条蝽': 21,
    '马尾松毛虫': 22,
    '麻皮蝽': 23,
    '黑蚱蝉': 24,
}


def getDataLoader(dirname):
    classes = os.listdir(dirname)

    for i, label in enumerate(classes):
        print(str(i) + ":" + "'" + label + "',")

    for i, label in enumerate(classes):
        print("'" + label + "'" + ":" + str(i) + ",")

# if __name__ == "__main__":
# getDataLoader(dirname='./data')
