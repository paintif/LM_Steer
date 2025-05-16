import torch
import torch.nn as nn


class Projected_Adaptor(nn.Module):
    """投影适配器类，用于在语言模型中实现可控的特征调整。
    
    该类提供了三种不同的适配方式：
    1. multiply: 使用低秩矩阵分解实现特征投影
    2. add: 直接在嵌入空间中添加向量
    3. offset: 在词汇表空间中添加偏移
    """
    
    def __init__(self, lm_head, adaptor_class, num_steers, embed_dim,
                 vocab_size, rank, epsilon, init_var, position="output",
                 hidden_dim=None, num_layers=3, dropout_rate=0.1):
        """初始化投影适配器。

        Args:
            lm_head: 语言模型的输出层
            adaptor_class: 适配器类型，可选值：'multiply', 'add', 'offset'
            num_steers: 控制向量的数量
            embed_dim: 嵌入维度
            vocab_size: 词汇表大小
            rank: 低秩矩阵的秩（仅用于multiply模式）
            epsilon: 适配强度的缩放因子
            init_var: 初始化参数的方差
            position: 适配器的位置，默认为'output'
        """
        super().__init__()
        assert rank > 0
        if adaptor_class == "multiply":
            # 使用两个投影矩阵实现低秩分解
            self.projector1 = nn.Parameter(torch.randn(
                num_steers, embed_dim, rank
            ) * init_var)
            self.projector2 = nn.Parameter(torch.randn(
                num_steers, embed_dim, rank
            ) * init_var)
        elif adaptor_class == "add":
            # 直接在嵌入空间添加向量
            self.add_vec = nn.Parameter(torch.randn(
                num_steers, embed_dim
            ))
        elif adaptor_class == "offset":
            # 在词汇表空间添加偏移
            self.offset_vec = nn.Parameter(torch.randn(
                num_steers, vocab_size
            ))
        elif adaptor_class == "nonlinear":
            # replace the linear layer with a multi-layer perceptron
            hidden_dim = hidden_dim or embed_dim
            self.steer_nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    *[nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    ) for _ in range(num_layers-2)],
                    nn.Linear(hidden_dim, embed_dim)
                )
                for _ in range(num_steers)
            ])
            self.bias = nn.Parameter(torch.zeros(num_steers, embed_dim))
        else:
            raise NotImplementedError()

        # 保存配置参数
        self.adaptor_class = adaptor_class
        self.rank = rank
        self.lm_head = lm_head
        self.epsilon = epsilon
        self.position = position
        self.num_steers = num_steers
        self.init_var = init_var
        self.steer_values = torch.zeros(num_steers)
        # 非线性的参数
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

    def set_value(self, steer_values):
        """设置控制向量的值。

        Args:
            steer_values: 控制向量的值，形状为(num_steers,)
        """
        self.steer_values = steer_values
    
    def forward(self, state):
        # 如果没有激活任何转向值，直接返回原始结果
        if self.steer_values.abs().sum() == 0:
            if self.position == "input":
                # 对于输入嵌入，直接返回state（因为它已经是嵌入索引）
                return self.lm_head(state)
            else:
                return state.matmul(
                    self.lm_head.weight.detach().transpose(0, 1))
                    
        # 获取 state 的批次大小
        batch_size = state.shape[0]
        
        # 将 self.steer_values 赋值给局部变量 steer_values
        steer_values_for_input = self.steer_values 
        
        # 检查 steer_values_for_input 是否是一维的
        if steer_values_for_input.dim() == 1:
            # 使其形状变为 [batch_size, num_steers]
            steer_values_for_input = steer_values_for_input.unsqueeze(0).expand(batch_size, -1)
        
        # 检查 steer_values 的形状是否与预期相符
        if steer_values_for_input.shape != (batch_size, self.num_steers):
            raise ValueError(f"steer_values_for_input should have shape [batch_size, num_steers], "
                             f"but got {steer_values_for_input.shape}")
        
        if self.position == "input":
            # 对于输入嵌入，我们需要先应用嵌入层，然后修改嵌入结果
            embedded = self.lm_head(state)  # 先获取嵌入
            
            if self.adaptor_class == "multiply":
                # 计算 delta
                delta = embedded[:, None].matmul(self.projector1[None]) * \
                    steer_values_for_input[:, :, None, None]
                
                delta = delta.matmul(
                    self.projector2.transpose(1, 2)[None]).sum(1)
                
                # 计算投影状态
                projected_embedded = embedded + self.epsilon * delta
                
                return projected_embedded
                
            elif self.adaptor_class == "add":
                add_values = steer_values_for_input.matmul(self.add_vec)
                projected_embedded = embedded + self.epsilon * add_values[:, None]
                return projected_embedded
                
            elif self.adaptor_class == "offset":
                # 对于offset，我们直接返回嵌入，因为offset应该应用于logits
                return embedded
        else:
            # 对于输出层，保持原有逻辑
            if self.adaptor_class == "multiply":
                # 计算 delta 的第一部分：
                delta = state[:, None].matmul(self.projector1[None]) *\
                    self.steer_values[:, :, None, None] 
                
                delta = delta.matmul(
                    self.projector2.transpose(1, 2)[None]).sum(1)
    
                # projected_state 的计算：将原始 state 加上经过 epsilon 缩放的 delta
                projected_state = state + self.epsilon * delta
                
                # 计算最终的 logits：将调整后的 projected_state 与语言模型的头部权重进行矩阵乘法
                logits = projected_state.matmul(
                    self.lm_head.weight.detach().transpose(0, 1))
                    
            elif self.adaptor_class == "add":
                add_values = self.steer_values.matmul(self.add_vec)
                projected_state = state + self.epsilon * add_values[:, None]
                logits = projected_state.matmul(
                    self.lm_head.weight.detach().transpose(0, 1))
                    
            elif self.adaptor_class == "offset":
                offset_values = self.steer_values.matmul(self.offset_vec)
                logits = state.matmul(
                    self.lm_head.weight.detach().transpose(0, 1))
                logits = logits + self.epsilon * offset_values[:, None]
            
            elif self.adaptor_class == "nonlinear":
                batch_size, seq_len, hidden_dim = state.shape
            
                # confirm steer_values is 2D tensor 
                if self.steer_values.dim() == 1:
                    # transform to 2D tensor
                    steer_values = self.steer_values.unsqueeze(0).expand(batch_size, -1)
                else:
                    # match the batch size
                    if self.steer_values.shape[0] != batch_size:
                        steer_values = self.steer_values[0].unsqueeze(0).expand(batch_size, -1)
                    else:
                        steer_values = self.steer_values
            
                # put state into steer_nets of each steer value
                delta = torch.zeros(batch_size, seq_len, self.num_steers, hidden_dim, device=state.device)
            
                for i, net in enumerate(self.steer_nets):
                    # shape the state for nets(batch_size*seq_len, hidden_dim)
                    flat_state = state.reshape(-1, hidden_dim)
                    transformed = net(flat_state)  # (batch_size*seq_len, hidden_dim)

                    # reshape back to original size(batch_size, seq_len, hidden_dim)
                    transformed = transformed.reshape(batch_size, seq_len, hidden_dim)

                    # calculate steer values' effect
                    delta[:, :, i, :] = transformed * steer_values[:, i, None, None]  # (B, L, D) * (B, 1, 1)

                # combinate steer values' effect
                bias = self.bias[None, None, :, :]  # (1, 1, S, D)
                delta = (delta + bias).sum(2)  # (B, L, D)

                # execute the projection
                projected_state = state + self.epsilon * delta
                logits = projected_state.matmul(
                    self.lm_head.weight.detach().transpose(0, 1))    
                
            return logits

    def regularization_term(self):
        """计算正则化项，用于防止适配器参数过大。

        Returns:
            float: 正则化项的值，为参数的L2范数平方和
        """
        if self.adaptor_class == "multiply":
            return self.projector1.pow(2).sum() + self.projector2.pow(2).sum()
        elif self.adaptor_class == "add":
            return self.add_vec.pow(2).sum()
        elif self.adaptor_class == "offset":
            return self.offset_vec.pow(2).sum()
        elif self.adaptor_class == "nonlinear":
            reg = self.bias.pow(2).sum()  # 修正：直接计算bias的平方和
            for net in self.steer_nets:
                for param in net.parameters():
                    reg += param.pow(2).sum()
            return reg

    def parameters(self):
        """获取适配器的可训练参数。

        Returns:
            list: 包含所有可训练参数的列表
        """
        if self.adaptor_class == "multiply":
            params = [self.projector1, self.projector2]
            return params
        elif self.adaptor_class == "add":
            return [self.add_vec]
        elif self.adaptor_class == "offset":
            return [self.offset_vec]
        elif self.adaptor_class == "nonlinear":
            params = [self.bias]  # 修正：直接将bias参数添加到列表中
            for net in self.steer_nets:
                params.extend(net.parameters())
            return params

    def state_dict(self):
        """获取适配器的状态字典，用于保存模型。

        Returns:
            dict: 包含适配器参数的状态字典
        """
        if self.adaptor_class == "multiply":
            state = {"projector1": self.projector1,
                    "projector2": self.projector2}
            return state
        elif self.adaptor_class == "add":
            return {"add_vec": self.add_vec}
        elif self.adaptor_class == "offset":
            return {"offset_vec": self.offset_vec}
        elif self.adaptor_class == "nonlinear":
            state_dict = {"bias": self.bias}
            for i, net in enumerate(self.steer_nets):
                state_dict[f"steer_net_{i}"] = net.state_dict()
            return state_dict

    def load_state_dict(self, state_dict):
        """从状态字典加载适配器参数。

        Args:
            state_dict: 包含适配器参数的状态字典
        """
        if self.adaptor_class == "multiply":
            self.projector1.data = state_dict["projector1"]
            self.projector2.data = state_dict["projector2"]
        elif self.adaptor_class == "add":
            self.add_vec.data = state_dict["add_vec"]
        elif self.adaptor_class == "offset":
            self.offset_vec.data = state_dict["offset_vec"]
        elif self.adaptor_class == "nonlinear":
            self.bias.data = state_dict["bias"]
            for i, net in enumerate(self.steer_nets):
                net.load_state_dict(state_dict[f"steer_net_{i}"])
