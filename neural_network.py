import numpy
import scipy.special


class NeuralNetWork:
    # 初始化
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # 设置每个输入、影藏、输出层中的节点数(三层神经网络)
        self.innodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        # 设置学习率
        self.lr = learning_rate

        # 权重矩阵
        # numpy.random.normal(loc, scale, size) 从正态分布中抽取随机样本
        # loc 分布的均值， scale 分布的标准差， size 输出值的维度。
        self.wih = numpy.random.normal(0.0, pow(self.innodes, -0.5), (self.hnodes, self.innodes))  # 输入层到隐藏层权重矩阵
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))  # 隐藏层到输出层权重矩阵

        # 创建激活函数
        self.activation_function = lambda x: scipy.special.expit(x)

    # 训练
    def train(self, inputs_list, targets_list):
        # 将输入列表转换成二维数组
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 将输入信号计算到隐藏层
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 计算隐藏层中输出的信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 将传输的信号计算到输出层
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 计算输出层中输出的信号
        final_outputs = self.activation_function(final_inputs)
        # 计算输出层的误差: target - final_outputs 预期目标输出值 - 实际计算得到的输出值
        output_errors = targets - final_outputs
        # 隐藏层的误差: 输出层误差按权重分割，在隐藏节点上重新组合
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 反向传播，更新各层权重(梯度下降法原理)
        # 更新隐藏层和输出层之间的权重
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        # 更新输入层和隐藏层之间的权重
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

    # 测试
    def query(self, inputs_list):
        # 将输入列表转换成二维数组
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 将输入信号计算到隐藏层
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 将信号从隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 将信号引入到输出层
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 将信号从输出层输出
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
