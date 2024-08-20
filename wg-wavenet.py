# Neural processing
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as lightning

# Dataset loading
from torch.utils.data import Dataset, DataLoader

# Audio processing
import scipy
import librosa
import librosa.filters
import numpy as np
from scipy.io import wavfile
import pickle

# Others
import multiprocessing as mp
import os

# Libraries versions
print(f"numpy: {np.__version__}")
print(f"torch: {torch.__version__}")
print(f"lightning: {lightning.__version__}")
print(f"scipy: {scipy.__version__}")
print(f"librosa: {librosa.__version__}")
print(f"CPU cores: {mp.cpu_count()}")

class Hyperparameters:
    """
    A class to hold hyperparameters for the model training and audio processing.
    """
    seed = 0
    ################################
    # Audio                        #
    ################################
    num_mels = 80  # Number of mel frequency bins
    num_freq = 513  # Number of frequency bins in the FFT
    sample_rate = 22050  # Sample rate for audio
    frame_shift = 200  # Shift between frames in milliseconds
    frame_length = 800  # Length of each frame in samples
    preemphasis = 0.97  # Preemphasis factor
    min_level_db = -100  # Minimum level in decibels
    ref_level_db = 20  # Reference level in decibels
    fmin = 0  # Minimum frequency for mel filter
    fmax = 8000  # Maximum frequency for mel filter
    segment_length = 16000  # Length of audio segments for processing

    train_dir = '../input/ljspeech11/LJSpeech-1.1/'  # Directory for training data

    ################################
    # Training                     #
    ################################
    pin_memory = True  # Whether to pin memory for DataLoader
    num_workers = mp.cpu_count()  # Number of workers for DataLoader
    prepare_data = False  # Flag to indicate if data preparation is needed
    checkpoint_path = None  # Path to save/load model checkpoints
    learning_rate = 4e-4  # Learning rate for the optimizer
    use_scheduler = True  # Flag to use learning rate scheduler
    scheduler_step = int(200e3)  # Step size for the scheduler
    scheduler_gamma = 0.5  # Multiplicative factor for learning rate decay
    scheduler_stop = int(800e3)  # Epoch at which to stop the scheduler
    accumulate_grad_batches = 1  # Number of batches to accumulate gradients
    batch_size = 16 // accumulate_grad_batches  # Effective batch size
    max_epochs = 610 * 2  # Maximum number of training epochs
    grad_norm = 10  # Gradient norm for clipping
    n = 3  # Number of iterations for logging
    iters_per_log = n * (10 // n)  # Iterations per log
    iters_per_sample = n * (500 // n)  # Iterations per sample
    iters_per_checkpoint = 10000  # Iterations per checkpoint

    ################################
    # Model                        #
    ################################
    up_scale = [2, 5, 2, 5, 2]  # Upsampling factors
    sigma = 0.6  # Noise level for the model
    num_flows = 4  # Number of flow layers in the model
    num_groups = 8  # Number of groups for audio processing
    num_layers = 7  # Number of layers in the model
    num_channels = 128  # Number of channels in the model
    kernel_size = 3  # Kernel size for convolutions
    PF_num_layers = 7  # Number of layers for the parallel flow
    PF_num_channels = 64  # Number of channels for the parallel flow

    ################################
    # Spectral Loss                #
    ################################
    mag_loss = True  # Flag to compute magnitude loss
    mel_loss = True  # Flag to compute mel loss
    fft_sizes = [2048, 1024, 512, 256, 128]  # FFT sizes for loss computation
    hop_sizes = [400, 200, 100, 50, 25]  # Hop sizes for STFT
    win_lengths = [2000, 1000, 500, 200, 100]  # Window lengths for STFT
    mel_scales = [4, 2, 1, 0.5, 0.25]  # Mel scale factors

# Seed for reproducibility
lightning.seed_everything(Hyperparameters.seed)

def load_wav(file_path, segment=True):
    """
    Load a WAV file and optionally segment it.

    Args:
        file_path (str): Path to the WAV file.
        segment (bool): Whether to segment the audio.

    Returns:
        np.ndarray: Normalized audio waveform.
    """
    sr, wav = wavfile.read(file_path)  # Read the WAV file
    wav = wav.astype(np.float32)  # Convert to float32
    wav = wav / np.max(np.abs(wav))  # Normalize the waveform
    try:
        assert sr == Hyperparameters.sample_rate  # Check sample rate
    except AssertionError:
        print('Error:', file_path, 'has wrong sample rate.')
    if not segment:
        return wav  # Return the full waveform if not segmenting
    if wav.shape[0] > Hyperparameters.segment_length:
        start = np.random.randint(0, len(wav) - Hyperparameters.segment_length)
        wav = wav[start:start + Hyperparameters.segment_length]  # Segment the waveform
    else:
        wav = np.pad(wav, (0, Hyperparameters.segment_length - wav.shape[0]), 'constant', constant_values=(0, 0))  # Pad if necessary
    return wav

def save_wav(wav, path):
    """
    Save a normalized waveform to a WAV file.

    Args:
        wav (np.ndarray): Normalized audio waveform.
        path (str): Path to save the WAV file.
    """
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))  # Scale to int16 range
    wavfile.write(path, Hyperparameters.sample_rate, wav.astype(np.int16))  # Write the WAV file

def preemphasis(x):
    """
    Apply preemphasis filter to the input signal.

    Args:
        x (np.ndarray): Input signal.

    Returns:
        np.ndarray: Preemphasized signal.
    """
    return scipy.signal.lfilter([1, -Hyperparameters.preemphasis], [1], x)

def inv_preemphasis(x):
    """
    Apply inverse preemphasis filter to the input signal.

    Args:
        x (np.ndarray): Input signal.

    Returns:
        np.ndarray: Deemphasized signal.
    """
    return scipy.signal.lfilter([1], [1, -Hyperparameters.preemphasis], x)

def spectrogram(y):
    """
    Compute the spectrogram of the input signal.

    Args:
        y (np.ndarray): Input signal.

    Returns:
        np.ndarray: Spectrogram.
    """
    D = _stft(preemphasis(y))
    S = _amp_to_db(np.abs(D)) - Hyperparameters.ref_level_db
    return _normalize(S)

def inv_spectrogram(spectrogram):
    """
    Reconstruct the waveform from the spectrogram using librosa.

    Args:
        spectrogram (np.ndarray): Spectrogram.

    Returns:
        np.ndarray: Reconstructed waveform.
    """
    S = _db_to_amp(_denormalize(spectrogram) + Hyperparameters.ref_level_db)  # Convert back to linear
    return inv_preemphasis(_griffin_lim(S ** Hyperparameters.power))  # Reconstruct phase

def melspectrogram(y):
    """
    Compute the mel spectrogram of the input signal.

    Args:
        y (np.ndarray): Input signal.

    Returns:
        np.ndarray: Mel spectrogram.
    """
    D = _stft(preemphasis(y))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - Hyperparameters.ref_level_db
    return _normalize(S)

def inv_melspectrogram(spectrogram):
    """
    Reconstruct the waveform from the mel spectrogram using librosa.

    Args:
        spectrogram (np.ndarray): Mel spectrogram.

    Returns:
        np.ndarray: Reconstructed waveform.
    """
    mel = _db_to_amp(_denormalize(spectrogram) + Hyperparameters.ref_level_db)
    S = _mel_to_linear(mel)
    return inv_preemphasis(_griffin_lim(S ** Hyperparameters.power))

def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
    """
    Find the endpoint of the audio signal based on silence detection.

    Args:
        wav (np.ndarray): Input waveform.
        threshold_db (float): Threshold in decibels for silence detection.
        min_silence_sec (float): Minimum duration of silence to consider as endpoint.

    Returns:
        int: Index of the endpoint.
    """
    window_length = int(Hyperparameters.sample_rate * min_silence_sec)
    hop_length = int(window_length / 4)
    threshold = _db_to_amp(threshold_db)
    for x in range(hop_length, len(wav) - window_length, hop_length):
        if np.max(wav[x:x+window_length]) < threshold:
            return x + hop_length
    return len(wav)

def _griffin_lim(S):
    """
    librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434

    Args:
        S (np.ndarray): Spectrogram.

    Returns:
        np.ndarray: Reconstructed waveform.
    """
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(Hyperparameters.gl_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y

def _stft(y):
    """
    Compute the Short-Time Fourier Transform (STFT) of the input signal.

    Args:
        y (np.ndarray): Input signal.

    Returns:
        np.ndarray: STFT of the input signal.
    """
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def _istft(y):
    """
    Compute the Inverse Short-Time Fourier Transform (ISTFT) of the input signal.

    Args:
        y (np.ndarray): Input signal.

    Returns:
        np.ndarray: ISTFT of the input signal.
    """
    _, hop_length, win_length = _stft_parameters()
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)

def _stft_parameters():
    """
    Get the parameters for STFT computation.

    Returns:
        tuple: (n_fft, hop_length, win_length)
    """
    return (Hyperparameters.num_freq - 1) * 2, Hyperparameters.frame_shift, Hyperparameters.frame_length

# Conversions:

_mel_basis = None

def _linear_to_mel(spectrogram):
    """
    Convert linear spectrogram to mel spectrogram.

    Args:
        spectrogram (np.ndarray): Linear spectrogram.

    Returns:
        np.ndarray: Mel spectrogram.
    """
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)

def _mel_to_linear(spectrogram):
    """
    Convert mel spectrogram to linear spectrogram.

    Args:
        spectrogram (np.ndarray): Mel spectrogram.

    Returns:
        np.ndarray: Linear spectrogram.
    """
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    inv_mel_basis = np.linalg.pinv(_mel_basis)
    inverse = np.dot(inv_mel_basis, spectrogram)
    inverse = np.maximum(1e-10, inverse)
    return inverse

def _build_mel_basis():
    """
    Build the mel filter basis.

    Returns:
        np.ndarray: Mel filter basis.
    """
    n_fft = (Hyperparameters.num_freq - 1) * 2
    return librosa.filters.mel(Hyperparameters.sample_rate, n_fft, n_mels=Hyperparameters.num_mels, fmin=Hyperparameters.fmin, fmax=Hyperparameters.fmax)

def _amp_to_db(x):
    """
    Convert amplitude to decibels.

    Args:
        x (np.ndarray): Amplitude.

    Returns:
        np.ndarray: Decibels.
    """
    return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
    """
    Convert decibels to amplitude.

    Args:
        x (np.ndarray): Decibels.

    Returns:
        np.ndarray: Amplitude.
    """
    return np.power(10.0, x * 0.05)

def _normalize(S):
    """
    Normalize the spectrogram.

    Args:
        S (np.ndarray): Spectrogram.

    Returns:
        np.ndarray: Normalized spectrogram.
    """
    return np.clip((S - Hyperparameters.min_level_db) / -Hyperparameters.min_level_db, 0, 1)

def _denormalize(S):
    """
    Denormalize the spectrogram.

    Args:
        S (np.ndarray): Normalized spectrogram.

    Returns:
        np.ndarray: Denormalized spectrogram.
    """
    return (np.clip(S, 0, 1) * -Hyperparameters.min_level_db) + Hyperparameters.min_level_db

def files_to_list(fdir):
    """
    Load the file list from the directory.

    Args:
        fdir (str): Directory containing the files.

    Returns:
        list: List of file paths or tuples of (waveform, mel spectrogram).
    """
    f_list = []
    with open(os.path.join(fdir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(fdir, 'wavs', '%s.wav' % parts[0])
            if Hyperparameters.prepare_data:
                wav = load_wav(wav_path, False)
                if wav.shape[0] < Hyperparameters.segment_length:
                    wav = np.pad(wav, (0, Hyperparameters.segment_length - wav.shape[0]), 'constant', constant_values=(0, 0))
                mel = melspectrogram(wav).astype(np.float32) 
                f_list.append([wav, mel])
            else:
                f_list.append(wav_path)
    if Hyperparameters.prepare_data and Hyperparameters.checkpoint_path is not None:
        with open(Hyperparameters.checkpoint_path, 'wb') as w:
            pickle.dump(f_list, w)
    return f_list

class LJDataset(Dataset):
    """
    Custom dataset class for loading LJSpeech dataset.

    Args:
        Dataset (torch.utils.data.Dataset): Base class for datasets.
    """
    def __init__(self, directory):
        """
        Initialize the dataset.

        Args:
            directory (str): Directory containing the dataset.
        """
        if Hyperparameters.prepare_data and Hyperparameters.checkpoint_path is not None and os.path.isfile(Hyperparameters.checkpoint_path):
            with open(Hyperparameters.checkpoint_path, 'rb') as r:
                self.file_list = pickle.load(r)  # Load preprocessed data from checkpoint
        else:
            self.file_list = files_to_list(directory)  # Load data from directory

    def __getitem__(self, index):
        """
        Get a data sample by index.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: (wav, mel) where wav is the audio waveform and mel is the mel spectrogram.
        """
        if Hyperparameters.prepare_data:
            wav, mel = self.file_list[index]
            segment_ml = Hyperparameters.segment_length // Hyperparameters.frame_shift + 1
            ms = np.random.randint(0, mel.shape[1] - segment_ml) if mel.shape[1] > segment_ml else 0
            ws = Hyperparameters.frame_shift * ms
            wav = wav[ws:ws + Hyperparameters.segment_length]  # Segment the waveform
            mel = mel[:, ms:ms + segment_ml]  # Segment the mel spectrogram
        else:
            wav = load_wav(self.file_list[index])  # Load the waveform
            mel = melspectrogram(wav).astype(np.float32)  # Compute mel spectrogram
        return wav, mel

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.file_list)

def collate_fn(batch):
    """
    Collate function for DataLoader.

    Args:
        batch (list): List of samples from the dataset.

    Returns:
        tuple: (wavs, mels) where wavs is a tensor of waveforms and mels is a tensor of mel spectrograms.
    """
    wavs = []
    mels = []
    for wav, mel in batch:
        wavs.append(wav)
        mels.append(mel)
    wavs = torch.Tensor(wavs)
    mels = torch.Tensor(mels)
    return wavs, mels

def mode(obj, model=False):
    """
    Move the object to the appropriate device (CPU or GPU).

    Args:
        obj (torch.Tensor or torch.nn.Module): Object to move to the device.
        model (bool): Whether the object is a model.

    Returns:
        torch.Tensor or torch.nn.Module: Object moved to the device.
    """
    d = torch.device('cuda' if Hyperparameters.is_cuda else 'cpu')
    return obj.to(d, non_blocking=False if model else Hyperparameters.pin_memory)

def to_arr(var):
    """
    Convert a PyTorch tensor to a NumPy array.

    Args:
        var (torch.Tensor): PyTorch tensor.

    Returns:
        np.ndarray: NumPy array.
    """
    return var.cpu().detach().numpy().astype(np.float32)

def prepare_dataloaders(fdir):
    """
    Prepare the dataloaders for training.

    Args:
        fdir (str): Directory containing the dataset.

    Returns:
        DataLoader: Dataloader for training.
    """
    trainset = LJDataset(fdir)
    train_loader = DataLoader(trainset, num_workers=Hyperparameters.num_workers, shuffle=True,
                             batch_size=Hyperparameters.batch_size, pin_memory=Hyperparameters.pin_memory,
                             drop_last=False, collate_fn=collate_fn)
    return train_loader

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    """
    Fused operation of adding, tanh, sigmoid, and multiplying.

    Args:
        input_a (torch.Tensor): First input tensor.
        input_b (torch.Tensor): Second input tensor.
        n_channels (torch.IntTensor): Number of channels.

    Returns:
        torch.Tensor: Result of the fused operation.
    """
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts

class Invertible1x1Conv(torch.nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """
    def __init__(self, c, device):
        """
        Initialize the Invertible1x1Conv layer.

        Args:
            c (int): Number of channels.
            device (torch.device): Device to use.
        """
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                    bias=False)
        self.device = device

        # Sample a random orthonormal matrix to initialize weights
        W = torch.linalg.qr(torch.FloatTensor(c, c).normal_())[0]

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:,0] = -1*W[:,0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z, reverse=False):
        """
        Forward pass of the Invertible1x1Conv layer.

        Args:
            z (torch.Tensor): Input tensor.
            reverse (bool): Whether to perform inverse convolution.

        Returns:
            torch.Tensor: Output tensor.
        """
        # shape
        batch_size, group_size, n_of_groups = z.size()

        W = self.conv.weight.squeeze()

        if reverse:
            if not hasattr(self, 'set'):
                # Reverse computation
                W_inverse = W.float().inverse()
                W_inverse = W_inverse[..., None]
                self.W_inverse = W_inverse
            z = torch.nn.functional.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            # Forward computation
            log_det_W = batch_size * n_of_groups * torch.logdet(W.cpu()).to(self.device)
            z = self.conv(z)
            return z, log_det_W

    def set_inverse(self):
        """
        Set the inverse of the weight matrix.
        """
        W = self.conv.weight.squeeze()
        W_inverse = W.float().inverse()
        W_inverse = W_inverse[..., None]
        self.W_inverse = W_inverse
        self.set = True

class WN(torch.nn.Module):
    """
    WaveNet-based model.
    """
    def __init__(self, n_in_channels, n_emb_channels, PF=False):
        """
        Initialize the WaveNet model.

        Args:
            n_in_channels (int): Number of input channels.
            n_emb_channels (int): Number of embedding channels.
            PF (bool): Whether to use parallel flow.
        """
        super(WN, self).__init__()
        assert(Hyperparameters.kernel_size % 2 == 1)
        assert(Hyperparameters.num_channels % 2 == 0)
        self.PF = PF
        self.num_layers = Hyperparameters.PF_num_layers if PF else Hyperparameters.num_layers
        self.num_channels = Hyperparameters.PF_num_channels if PF else Hyperparameters.num_channels
        self.in_layers = torch.nn.ModuleList()
        self.cond_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(n_in_channels, self.num_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        if PF:
            self.end = torch.nn.Conv1d(self.num_channels, n_in_channels, 1)
        else:
            self.end = torch.nn.Conv1d(self.num_channels, 2*n_in_channels, 1)
            self.end.weight.data.zero_()
            self.end.bias.data.zero_()

        for i in range(self.num_layers):
            dilation = 2**i if PF else 2**i
            padding = int((Hyperparameters.kernel_size*dilation - dilation)/2)
            in_layer = torch.nn.Conv1d(self.num_channels, 2*self.num_channels, Hyperparameters.kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            cond_layer = torch.nn.Conv1d(n_emb_channels, 2*self.num_channels, 1)
            cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
            self.cond_layers.append(cond_layer)

            # last one is not necessary
            if i < self.num_layers - 1:
                res_skip_channels = 2*self.num_channels
            else:
                res_skip_channels = self.num_channels
            res_skip_layer = torch.nn.Conv1d(self.num_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, wavs, mels):
        """
        Forward pass of the WaveNet model.

        Args:
            wavs (torch.Tensor): Input waveforms.
            mels (torch.Tensor): Input mel spectrograms.

        Returns:
            torch.Tensor: Output waveforms.
        """
        wavs = self.start(wavs)

        for i in range(self.num_layers):
            acts = fused_add_tanh_sigmoid_multiply(
                self.in_layers[i](wavs),
                self.cond_layers[i](mels),
                torch.IntTensor([self.num_channels]))

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.num_layers - 1:
                wavs = res_skip_acts[:,:self.num_channels,:] + wavs
                skip_acts = res_skip_acts[:,self.num_channels:,:]
            else:
                skip_acts = res_skip_acts

            if i == 0:
                output = skip_acts
            else:
                output = skip_acts + output
        return self.end(torch.nn.ReLU()(output) if self.PF else output)

class US(torch.nn.Module):
    """
    Upsampling module.
    """
    def __init__(self):
        """
        Initialize the Upsampling module.
        """
        super(US, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Conv1d(Hyperparameters.num_mels, Hyperparameters.num_mels, kernel_size=2))
        self.layers.append(torch.nn.ReLU())
        for sf in Hyperparameters.up_scale:
            self.layers.append(torch.nn.Upsample(scale_factor=sf))
            self.layers.append(torch.nn.Conv1d(Hyperparameters.num_mels, Hyperparameters.num_mels,
                                              kernel_size=5, padding=2))
            self.layers.append(torch.nn.ReLU())

    def forward(self, mels):
        """
        Forward pass of the Upsampling module.

        Args:
            mels (torch.Tensor): Input mel spectrograms.

        Returns:
            torch.Tensor: Upsampled mel spectrograms.
        """
        for f in self.layers:
            mels = f(mels)
        return mels

class Model(torch.nn.Module):
    """
    WG-WaveNet model.
    """
    def __init__(self, device):
        """
        Initialize the WG-WaveNet model.

        Args:
            device (torch.device): Device to use.
        """
        super(Model, self).__init__()
        assert(Hyperparameters.num_groups % 2 == 0)
        self.num_groups = Hyperparameters.num_groups
        self.num_flows = Hyperparameters.num_flows
        self.upsample = US()
        self.WN = WN(int(Hyperparameters.num_groups/2), Hyperparameters.num_mels*Hyperparameters.num_groups)
        self.convinv = torch.nn.ModuleList()
        self.device = device
        for k in range(Hyperparameters.num_flows):
            self.convinv.append(Invertible1x1Conv(Hyperparameters.num_groups, device))
        self.PF = WN(1, Hyperparameters.num_mels, True)

    def forward(self, wavs, mels):
        """
        Forward pass of the WG-WaveNet model.

        Args:
            wavs (torch.Tensor): Input waveforms.
            mels (torch.Tensor): Input mel spectrograms.

        Returns:
            tuple: (z, log_s_list, log_det_W_list)
        """
        # Upsample spectrogram to size of audio
        mels = self.upsample(mels)
        assert(mels.size(2) == wavs.size(1))

        mels = mels.unfold(2, self.num_groups, self.num_groups).permute(0, 2, 1, 3) # (batch_size, seg_l//num_groups, num_mels, num_groups)
        mels = mels.contiguous().view(mels.size(0), mels.size(1), -1).permute(0, 2, 1) # (batch_size, num_mels*num_groups, seg_l//num_groups)

        audio = wavs.unfold(1, self.num_groups, self.num_groups).permute(0, 2, 1) # (batch_size, num_groups, seg_l//num_groups)

        log_s_list = []
        log_det_W_list = []

        for k in range(self.num_flows):
            audio, log_det_W = self.convinv[k](audio)
            log_det_W_list.append(log_det_W)

            n_half = int(audio.size(1)/2)
            audio_0 = audio[:,:n_half,:]
            audio_1 = audio[:,n_half:,:]

            output = self.WN(audio_0, mels).clamp(-10, 10)
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = torch.exp(log_s)*audio_1+b
            log_s_list.append(log_s)

            audio = torch.cat([audio_0, audio_1],1).clamp(-10, 10)

        return audio, log_s_list, log_det_W_list

    def WG(self, inp_mels):
        """
        Generate waveforms using the WaveNet model.

        Args:
            inp_mels (torch.Tensor): Input mel spectrograms.

        Returns:
            torch.Tensor: Generated waveforms.
        """
        # (batch_size, T//num_groups, num_mels, num_groups)
        mels = inp_mels.unfold(2, self.num_groups, self.num_groups).permute(0, 2, 1, 3)
        # (batch_size, num_mels*num_groups, T//num_groups)
        mels = mels.contiguous().view(mels.size(0), mels.size(1), -1).permute(0, 2, 1)

        audio = torch.FloatTensor(mels.size(0),
                                 Hyperparameters.num_groups,
                                 mels.size(2)).normal_().cuda()

        audio = Hyperparameters.sigma * audio

        for k in reversed(range(self.num_flows)):
            n_half = int(audio.size(1)/2)
            audio_0 = audio[:,:n_half,:]
            audio_1 = audio[:,n_half:,:]

            output = self.WN(audio_0, mels)
            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b)/torch.exp(s)
            audio = torch.cat([audio_0, audio_1],1)

            audio = self.convinv[k](audio, reverse=True)

        audio = audio.permute(0, 2, 1).contiguous().view(audio.size(0), 1, -1) # (batch_size, 1, seg_l)
        return audio

    def infer(self, mels):
        """
        Generate waveforms from mel spectrograms.

        Args:
            mels (torch.Tensor): Input mel spectrograms.

        Returns:
            torch.Tensor: Generated waveforms.
        """
        inp_mels = self.upsample(mels) # (batch_size, num_mels, T)
        audio = self.WG(inp_mels)
        d = inp_mels.size(2)-audio.size(2)
        if d > 0:
            audio = torch.cat([audio, 0*audio[:, :, :d]], 2)
        audio = self.PF(audio, inp_mels).squeeze(1)
        return audio

    def set_inverse(self):
        """
        Set the inverse of the weight matrices.
        """
        for i in range(Hyperparameters.num_flows):
            self.convinv[i].set_inverse()

    @staticmethod
    def remove_weightnorm(model):
        """
        Remove weight normalization from the model.

        Args:
            model (torch.nn.Module): Model to remove weight normalization from.

        Returns:
            torch.nn.Module: Model without weight normalization.
        """
        waveglow = model
        for WN in [waveglow.WN, waveglow.PF]:
            WN.start = torch.nn.utils.remove_weight_norm(WN.start)
            WN.in_layers = remove(WN.in_layers)
            WN.cond_layers = remove(WN.cond_layers)
            WN.res_skip_layers = remove(WN.res_skip_layers)
        return waveglow

def remove(conv_list):
    """
    Remove weight normalization from a list of convolutional layers.

    Args:
        conv_list (torch.nn.ModuleList): List of convolutional layers.

    Returns:
        torch.nn.ModuleList: List of convolutional layers without weight normalization.
    """
    new_conv_list = torch.nn.ModuleList()
    for old_conv in conv_list:
        old_conv = torch.nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
    return new_conv_list

class Loss(torch.nn.Module):
    """
    Loss function for the WG-WaveNet model.
    """
    def __init__(self, device):
        """
        Initialize the Loss function.

        Args:
            device (torch.device): Device to use.
        """
        super(Loss, self).__init__()
        self.d = 2*Hyperparameters.sigma*Hyperparameters.sigma
        self.loss = MultiResolutionSTFTLoss(device, Hyperparameters.fft_sizes, Hyperparameters.hop_sizes,
                                            Hyperparameters.win_lengths, Hyperparameters.mel_scales)

    def forward(self, model_output, p_wavs=None, r_wavs=None):
        """
        Forward pass of the Loss function.

        Args:
            model_output (tuple): Output of the WG-WaveNet model.
            p_wavs (torch.Tensor): Predicted waveforms.
            r_wavs (torch.Tensor): Reference waveforms.

        Returns:
            tuple: (total_loss, zloss, sloss)
        """
        # zloss
        z, log_s_list, log_w_list = model_output
        log_s_total = 0
        log_w_total = 0
        for i, log_s in enumerate(log_s_list):
            log_s_total += torch.sum(log_s)
            log_w_total += torch.sum(log_w_list[i])
        zloss = torch.sum(z*z)/self.d-log_s_total-log_w_total
        zloss /= (z.size(0)*z.size(1)*z.size(2))

        # sloss
        sloss = self.loss(p_wavs, r_wavs) if p_wavs is not None else 0*zloss

        return zloss+sloss, zloss, sloss

class MultiResolutionSTFTLoss(torch.nn.Module):
    """
    Multi-resolution STFT loss module.
    """
    def __init__(self, device,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 mel_scales=[1, 1, 1],
                 window="hann_window"):
        """
        Initialize the MultiResolutionSTFTLoss module.

        Args:
            device (torch.device): Device to use.
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            mel_scales (list): List of mel scales.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        self.bases = []
        for fs, ss, wl, sc in zip(fft_sizes, hop_sizes, win_lengths, mel_scales):
            self.stft_losses += [STFTLoss(device, fs, ss, wl, window)]
            b = librosa.filters.mel(Hyperparameters.sample_rate, fs, n_mels=Hyperparameters.num_mels*sc, fmax=Hyperparameters.fmax).T
            self.bases += [torch.Tensor(b, device=device)]

    def forward(self, x, y):
        """
        Forward pass of the MultiResolutionSTFTLoss module.

        Args:
            x (torch.Tensor): Predicted waveforms.
            y (torch.Tensor): Reference waveforms.

        Returns:
            tuple: (sc_loss, spec_loss)
        """
        sc_loss = 0.0
        spec_loss = 0.0
        for f, b in zip(self.stft_losses, self.bases):
            sc_l, spec_l = f(x, y, b)
            sc_loss += sc_l
            spec_loss += spec_l
        sc_loss /= len(self.stft_losses)
        spec_loss /= len(self.stft_losses)

        return sc_loss+spec_loss

class STFTLoss(torch.nn.Module):
    """
    STFT loss module.
    """
    def __init__(self, device, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """
        Initialize the STFTLoss module.

        Args:
            device (torch.device): Device to use.
            fft_size (int): FFT size.
            shift_size (int): Hop size.
            win_length (int): Window length.
            window (str): Window function type.
        """
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length, device=device).cuda()

    def forward(self, x, y, b):
        """
        Forward pass of the STFTLoss module.

        Args:
            x (torch.Tensor): Predicted waveforms.
            y (torch.Tensor): Reference waveforms.
            b (torch.Tensor): Mel basis.

        Returns:
            tuple: (sc_loss, spec_loss)
        """
        x_mag, x_mel = stft(x, self.fft_size, self.shift_size, self.win_length, self.window, b)
        y_mag, y_mel = stft(y, self.fft_size, self.shift_size, self.win_length, self.window, b)
        sc_loss = spec_loss = 0
        if Hyperparameters.mag_loss:
            h = x_mag.size(2)*2*Hyperparameters.fmax//Hyperparameters.sample_rate if Hyperparameters.sample_rate >= 2*Hyperparameters.fmax else x_mag.size(2)
            x_mag_ = x_mag[:, :, :h]
            y_mag_ = y_mag[:, :, :h]
            sc_loss += torch.norm((y_mag_-x_mag_), p="fro")/torch.norm(y_mag_, p="fro")
            spec_loss += torch.nn.L1Loss()(torch.log(x_mag_), torch.log(y_mag_))
            if h < x_mag.size(2):
                x_mag_m = x_mag[:, :, h:].mean(1)
                y_mag_m = y_mag[:, :, h:].mean(1)
                sc_loss += torch.norm((y_mag_m-x_mag_m), p="fro")/torch.norm(y_mag_m, p="fro")
                spec_loss += torch.nn.L1Loss()(torch.log(x_mag_m), torch.log(y_mag_m))
        if Hyperparameters.mel_loss:
            sc_loss += torch.norm((y_mel-x_mel), p="fro")/torch.norm(y_mel, p="fro")
            spec_loss += torch.nn.L1Loss()(torch.log(x_mel), torch.log(y_mel))
        s = int(Hyperparameters.mag_loss)+int(Hyperparameters.mel_loss)
        if s == 0:
            print('Error: Hyperparameters.mag_loss and Hyperparameters.mel_loss are both set as False.')
            exit()
        return sc_loss/s, spec_loss/s

def stft(x, fft_size, hop_size, win_length, window, b):
    """
    Perform STFT and convert to magnitude spectrogram.

    Args:
        x (torch.Tensor): Input waveforms.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (torch.Tensor): Window function.
        b (torch.Tensor): Mel basis.

    Returns:
        tuple: (mag, mel)
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    mag = torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)
    return mag, torch.clamp(torch.matmul(mag.cuda(), b.cuda()), min=1e-7**0.5)

train_dataloader = prepare_dataloaders(Hyperparameters.train_dir)
print(f"Total batches: {len(train_dataloader)}")

current_epoch = 309

class WGWaveNetAlgorithm(lightning.LightningModule):
    """
    Lightning module for the WG-WaveNet algorithm.

    Args:
        lightning.LightningModule: Base class for PyTorch Lightning modules.
    """
    def __init__(self):
        """
        Initialize the WG-WaveNet algorithm.
        """
        super().__init__()
        self.model = Model(self.device)  # Initialize the model
        self.criterion = Loss(self.device)  # Initialize the loss function

    def forward(self, batch):
        """
        Forward pass for the model.

        Args:
            batch (tuple): Input batch of data.

        Returns:
            tuple: Output from the model.
        """
        return batch

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (tuple): Input batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Total loss for the batch.
        """
        wavs, mels = batch  # Unpack the batch
        outputs = self.model(wavs, mels)  # Forward pass through the model
        p_wavs = self.model.infer(mels) if batch_idx % Hyperparameters.n == 0 else None  # Infer previous waveforms if needed
        total_loss, zloss, sloss = self.criterion(outputs, p_wavs, wavs)  # Compute the loss
        return total_loss  # Return the total loss

    def configure_optimizers(self):
        """
        Configure the optimizers and learning rate scheduler.

        Returns:
            dict: Dictionary containing the optimizer and scheduler.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=Hyperparameters.learning_rate)  # Initialize optimizer
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, Hyperparameters.scheduler_step, Hyperparameters.scheduler_gamma, last_epoch=-1)  # Initialize scheduler
        
        for _ in range(current_epoch):
            lr_scheduler.step()  # Step the scheduler
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }

if __name__ == '__main__':
    """
    Main entry point for training the WG-WaveNet model.
    """
    wgwavenet_algorithm = WGWaveNetAlgorithm()  # Initialize the algorithm
    training_session = 5  # Number of training sessions
    ckpt_path = f"../input/wg-wavenet-lightning-v2/lightning_logs/version_0/checkpoints/epoch={current_epoch}-step=253384.ckpt"  # Path to checkpoint

    trainer = lightning.Trainer(
        max_time={'hours': 11 * training_session, 'minutes': 50 * training_session},  # Maximum training time
        max_epochs=Hyperparameters.max_epochs,  # Maximum number of epochs
        accumulate_grad_batches=Hyperparameters.accumulate_grad_batches,  # Gradient accumulation
        precision=16,  # Precision for training
        accelerator="auto",  # Automatically select accelerator
        devices=-1,  # Use all available devices
        reload_dataloaders_every_n_epochs=1,  # Reload dataloaders every epoch
    )

    # Fit the model to the dataset.
    trainer.fit(
        wgwavenet_algorithm, 
        train_dataloader,
        ckpt_path=ckpt_path  # Load from checkpoint
    )