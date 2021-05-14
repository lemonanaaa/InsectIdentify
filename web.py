from flask import Flask, request, jsonify
from getModel import get_myTrained_model
import torch
import random
import string
import predict

app = Flask(__name__)
basedir = './log/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = get_myTrained_model('./model/myModel-0.9.pkl')


@app.route('/', methods=['post'])
def predictClass():
    # 生成随机字符串，防止图片名字重复
    ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 16))
    # 获取图片文件 name = picture
    img = request.files.get('picture')  # 方法一 从客户端获取数据
    print(type(img))
    # 定义一个图片存放的位置 存放在log下面
    path = basedir
    # 图片名称 给图片重命名 为了图片名称的唯一性
    imgName = ran_str + img.filename
    # 图片path和名称组成图片的保存路径
    file_path = path + imgName
    # 保存图片
    img.save(file_path)
    pre_labels_id, pre_labels_name = predict.predict(image_path=file_path, device=device, model=model)

    res_json = dict()
    res_json['pre_labels_id'] = str(pre_labels_id)
    res_json['pre_labels_name'] = str(pre_labels_name)
    return jsonify(res_json)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
