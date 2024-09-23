import torch.nn as nn
from torch.optim import Adam


class CNN(nn.Module):
    def __init__(self, num_classes, in_channels=1, features_fore_linear=64*6*6, lr=0.001):
        super().__init__()
        # Layer structure pulled from https://github.com/abhishekyana/speech-enhancement-with-cnns/blob/master/INFERING/infer.py
        block1 = [
            nn.Conv2d(1, 18, kernel_size=(9, 8), padding=(4, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Conv2d(18, 30, kernel_size=[5, 1], padding=((5-1)//2, 0), bias=False),
        ]
        block2 = [
            nn.ReLU(),
            nn.BatchNorm2d(30),
            nn.Conv2d(30, 8, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 18, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Conv2d(18, 30, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
        ]
        block3 = [
            nn.ReLU(),
            nn.BatchNorm2d(30),
            nn.Conv2d(30, 8, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 18, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Conv2d(18, 30, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(30),
            nn.Conv2d(30, 8, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 18, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Conv2d(18, 30, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
        ]
        block4 = [
            nn.ReLU(),
            nn.BatchNorm2d(30),
            nn.Conv2d(30, 8, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 18, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Conv2d(18, 30, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
        ]
        block5 = [
            nn.ReLU(),
            nn.BatchNorm2d(30),
            nn.Conv2d(30, 8, kernel_size=[9, 1], padding=((9-1)//2, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(8, 1, kernel_size=[129, 1], padding=((129-1)//2, 0), bias=False),
        ]
        self.block1 = nn.Sequential(*block1)
        self.block2 = nn.Sequential(*block2)
        self.block3 = nn.Sequential(*block3)
        self.block4 = nn.Sequential(*block4)
        self.block5 = nn.Sequential(*block5)
    
    def forward(self, x):
        skip0 = self.block1(x)
        skip1 = self.block2(skip0)
        out = self.block3(skip1)
        out = self.block4(out + skip1)
        out = self.block5(out + skip0)
        return out

def make_input_windows(stft_features, num_segments=8, num_features=129):
    noisy_stft = np.concatenate([stft_features[:, 0:num_segments - 1], stft_features], axis=1)
    stft_segments = np.zeros((num_features, num_segments, noisy_stft.shape[1] - num_segments + 1))

    for i in range(noisy_stft.shape[1] - num_segments + 1):
        stft_segments[:, :, i] = noisy_stft[:, i:i + num_segments]
    return stft_segments

def get_stft(audio):
    return librosa.stft(audio, 
                        n_fft=FFT_LENGTH, 
                        win_length=WINDOW_LENGTH, 
                        hop_length=OVERLAP, 
                        window=scipy.signal.hamming(256, sym=False),
                        center=True)


def make_input_windows(stft_features, num_segments=8, num_features=129):
    noisy_stft = np.concatenate([stft_features[:, 0:num_segments - 1], stft_features], axis=1)
    stft_segments = np.zeros((num_features, num_segments, noisy_stft.shape[1] - num_segments + 1))

    for i in range(noisy_stft.shape[1] - num_segments + 1):
        stft_segments[:, :, i] = noisy_stft[:, i:i + num_segments]
    return stft_segments

def stft_to_audio(features, phase, window_length, overlap):
    features = np.squeeze(features)
    features = features * np.exp(1j * phase)
    features = features.transpose(1, 0)
    return librosa.istft(features, win_length=window_length, hop_length=overlap)


def clean_audio_waveform(testing_audio, mymodel, cuda=False, msize=2**9):
    testing_audio_stft = get_stft(testing_audio)
    testing_audio_mag, testing_audio_phase = np.abs(testing_audio_stft), np.angle(testing_audio_stft)
    testing_audio_input_windows = make_input_windows(testing_audio_mag)
    fs, ss, m = testing_audio_input_windows.shape
    Tmp = []
    for i in tqdm(range(0, m, msize)):
        testing_tensor = torch.Tensor(testing_audio_input_windows[:, :, i:i+msize]).permute(2, 0, 1)
        if cuda and torch.cuda.is_available():
            testing_tensor = testing_tensor.cuda()
        testing_prediction = mymodel(testing_tensor.unsqueeze(1))
        clean_testing = testing_prediction.squeeze().cpu().detach().numpy()
        clean_testing_audio = stft_to_audio(clean_testing, testing_audio_phase[:, i:i+msize].T, WINDOW_LENGTH, OVERLAP)
        Tmp.append(clean_testing_audio)
    return np.concatenate(Tmp)

    # def train_model(self, train_dataloader, epochs=1, val_dataloader=None):
        
    #     # To hold accuracy during training and testing
    #     train_accs = []
    #     test_accs = []

    #     for epoch in range(epochs):
            
    #         epoch_acc = 0

    #         for inputs, targets in tqdm(train_dataloader):
    #             logits = self(inputs)
    #             loss = self.criterion(logits, targets)
    #             loss.backward()

    #             self.optim.step()
    #             self.optim.zero_grad()

    #             # Not actually used for training, just for keeping track of accuracy
    #             epoch_acc += (torch.argmax(logits, dim=1) == targets).sum().item()

    #         train_accs.append(epoch_acc / len(train_dataloader.dataset))

    #         # If we have val dataloader, we can evaluate after each epoch
    #         if val_dataloader is not None:
    #             acc = self.eval_model(val_dataloader)
    #             test_accs.append(acc)
    #             print(f"Epoch {epoch} validation accuracy: {acc}")
        
    #     return train_accs, test_accs

    # def eval_model(self, test_dataloader):
        
    #     total_acc = 0

    #     for input_batch, label_batch in test_dataloader:
    #         # Get predictions
    #         logits = self(input_batch)

    #         # Remember, outs are probabilities (so there's 10 for each input)
    #         # The classification the network wants to assign, must therefore be the probability with the larget value
    #         # We find that using argmax (dim=1, because dim=0 would be across batch dimension)
    #         classifications = torch.argmax(logits, dim=1)
    #         total_acc += (classifications == label_batch).sum().item()

    #     total_acc = total_acc / len(test_dataloader.dataset)

    #     return total_acc