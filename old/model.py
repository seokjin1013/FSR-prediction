import torch

__init__ = [
    'LSTM',
    'CNN_LSTM',
]

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


# class CNN(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, num_layer, output_size):
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layer = num_layer
#         self.output_size = output_size
#         self.encoder = torch.nn.Conv1d()