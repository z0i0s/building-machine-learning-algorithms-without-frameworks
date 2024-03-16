import numpy as np

def softmax(x):
    # 计算softmax函数
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    # 实现缩放点积注意力机制
    matmul_qk = np.matmul(Q, K.transpose(-1, -2))  # 执行Q和K的矩阵乘法
    dk = K.shape[-1]  # 获取键向量的维度
    scaled_attention_logits = matmul_qk / np.sqrt(dk)  # 缩放点积
    attention_weights = softmax(scaled_attention_logits)  # 应用softmax得到注意力权重
    output = np.matmul(attention_weights, V)  # 将注意力权重和V值相乘得到最终的输出
    return output, attention_weights

class FeedForwardNetwork:
    # 简单的前馈网络
    def __init__(self, d_model, d_ff):
        # 初始化权重和偏置
        self.W1 = np.random.randn(d_model, d_ff)
        self.b1 = np.zeros((1, d_ff))
        self.W2 = np.random.randn(d_ff, d_model)
        self.b2 = np.zeros((1, d_model))

    def forward(self, x):
        # 前向传播
        x = np.dot(x, self.W1) + self.b1  # 第一层线性变换
        x = np.maximum(0, x)  # ReLU激活函数
        x = np.dot(x, self.W2) + self.b2  # 第二层线性变换
        return x

class SimpleTransformer:
    def __init__(self, input_dim, d_model, output_dim):
        # 初始化Transformer参数
        self.d_model = d_model
        self.Wq = np.random.randn(input_dim, d_model)
        self.Wk = np.random.randn(input_dim, d_model)
        self.Wv = np.random.randn(input_dim, d_model)
        self.Wo = np.random.randn(d_model, output_dim)
        self.ffn = FeedForwardNetwork(d_model, d_model * 4)  # 假设前馈网络隐藏层是模型维度的4倍

    def forward(self, x):
        # Transformer前向传播
        Q = np.dot(x, self.Wq)
        K = np.dot(x, self.Wk)
        V = np.dot(x, self.Wv)
        attention_output, _ = scaled_dot_product_attention(Q, K, V)  # 应用自注意力机制
        ffn_output = self.ffn.forward(attention_output)  # 通过前馈网络
        output = np.dot(ffn_output, self.Wo)  # 线性层
        return output

    def train(self, X, Y, epochs=10, learning_rate=0.01):
        # 简单的训练循环
        m = X.shape[0]  # 获取样本数
        for epoch in range(epochs):
            preds = self.forward(X)  # 前向传播
            loss = np.mean((preds - Y) ** 2)  # 计算均方误差损失

            # 这里省略了反向传播和参数更新步骤，因为需要实现完整的梯度计算
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')  # 每10个epoch打印一次损失

    def predict(self, X):
        # 预测函数
        preds = self.forward(X)  # 使用前向传播进行预测
        return preds

# 示例用法
input_dim = 4  # 输入特征维度
d_model = 64  # 模型维度
output_dim = 1  # 输出维度
seq_length = 10  # 序列长度
m = 5  # 批量大小

np.random.seed(42)
X = np.random.randn(m, seq_length, input_dim)  # 随机生成输入数据
Y = np.random.randn(m, seq_length, output_dim)  # 随机生成目标数据

transformer = SimpleTransformer(input_dim, d_model, output_dim)
transformer.train(X, Y, epochs=100, learning_rate=0.001)  # 训练模型
