import torch

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.output_size = output_size
        self.encoder = torch.nn.LSTM(input_size, hidden_size, num_layer)
        self.decoder = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x, _ = self.encoder(x)
        x = self.decoder(x)
        return x


class CNN_LSTM(torch.nn.Module):
    def __init__(self, input_size, cnn_hidden_size, lstm_hidden_size, cnn_num_layer, lstm_num_layer, output_size):
        super().__init__()
        self.input_size = input_size
        self.cnn_hidden_size = cnn_hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.cnn_num_layer = cnn_num_layer
        self.lstm_num_layer = lstm_num_layer
        self.output_size = output_size
        self.cnn_encoder = torch.nn.Sequential()
        self.cnn_encoder.append(torch.nn.Conv1d(input_size, cnn_hidden_size, 3, padding=1))
        self.cnn_encoder.extend(torch.nn.Conv1d(cnn_hidden_size, cnn_hidden_size, 3, padding=1) for _ in range(cnn_num_layer - 1))
        self.lstm_encoder = torch.nn.LSTM(cnn_hidden_size, lstm_hidden_size, lstm_num_layer)
        self.decoder = torch.nn.Linear(lstm_hidden_size, output_size)
    
    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.cnn_encoder(x)
        x = x.transpose(-1, -2)
        x, _ = self.lstm_encoder(x)
        x = self.decoder(x)
        return x


class ANN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.output_size = output_size
        layers = [[hidden_size, hidden_size] for i in range(num_layer)]
        layers[0][0] = input_size
        layers[-1][1] = output_size
        self.layers = torch.nn.Sequential(*[
            torch.nn.Linear(layer[0], layer[1]) for layer in layers
        ])
    
    def forward(self, x):
        x = self.layers(x)
        return x


class Attention(torch.nn.Module):
    def __init__(self, input_size, output_size, num_heads=4):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.attention = torch.nn.MultiheadAttention(input_size, num_heads)
        self.decoder = torch.nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x, _ = self.attention(x, x, x)
        x = self.decoder(x)
        return x


class FSRGraphNeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, num_layer, output_size):
        super().__init__()
        self.input_size = input_size
        self.num_layer = num_layer
        self.output_size = output_size
        layers = [[input_size, input_size] for i in range(num_layer)]
        layers[-1][1] = output_size
        self.weights = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(input_size, output_size)) for input_size, output_size in layers
        ])
        self.bias = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(output_size)) for _, output_size in layers
        ])
        for w in self.weights:
            torch.nn.init.kaiming_uniform_(w)
        for b in self.bias:
            torch.nn.init.zeros_(b)
        self.adj_matrix = None
        if input_size == 6:
            self.adj_matrix = [
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0, 1],
                [0, 1, 1, 0, 1, 0],
                [0, 1, 1, 1, 0, 1],
            ]
        elif input_size == 12:
            self.adj_matrix = [
                [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
                [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
                [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
            ]
        if not self.adj_matrix:
            assert 'input_size is incompatible'
        self.adj_matrix = torch.tensor(self.adj_matrix).float()
    
    def forward(self, x):
        for w, b in zip(self.weights, self.bias):
            x = x.matmul(self.adj_matrix).matmul(w) + b
        return x
        