from flask import Flask
from flask import request
from flask import jsonify
import numpy

from image import get_pic_array
from neural_network import NeuralNetWork


app = Flask(__name__)


input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.1


# 初始化神经网络
b = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)


def traning():
    # 读入训练数据
    training_data_file = open("mnist_train.csv", "r")
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    count = 0
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.9) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        b.train(inputs, targets)
        count += 1
        print("已训练数据：", count)
    print("训练完成")


@app.route('/', methods=["GET"])
def query():
    filename = request.args.get("filename", "")
    data = get_pic_array(filename)
    inputs = (numpy.asfarray(data) / 255.0 * 0.99) + 0.01
    outputs = b.query(inputs)
    label = numpy.argmax(outputs)
    return jsonify({"code": 200, "data": int(label), "msg": "成功"})


@app.route('/test', methods=["GET"])
def test():
    # 使用测试数据测试准确性
    test_data_file = open("mnist_test.csv", "r")
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    # 逐项对比测试数据是否准确，进行统计
    scorecard = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = b.query(inputs)
        label = numpy.argmax(outputs)
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    scorecard_array = numpy.asarray(scorecard)
    return jsonify({"code": 200, "data": f"准确率:{scorecard_array.sum() / scorecard_array.size}", "msg": "成功"})


if __name__ == '__main__':
    traning()
    app.run()
