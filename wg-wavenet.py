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

class hparams:
	seed = 0
	################################
	# Audio                        #
	################################
	num_mels = 80
	num_freq = 513
	sample_rate = 22050
	frame_shift = 200
	frame_length = 800
	preemphasis = 0.97
	min_level_db = -100
	ref_level_db = 20
	fmin = 0
	fmax = 8000
	seg_l = 16000

	train_dir = '../input/ljspeech11/LJSpeech-1.1/'

	################################
	# Train	                       #
	################################
	# is_cuda = True
	pin_mem = True
	n_workers = mp.cpu_count()
	prep = False
	pth = None
	lr = 4e-4
	sch = True
	sch_step = int(200e3)
	sch_g = 0.5
	sch_stop = int(800e3)
	# max_iter = int(1000e3)
	accumulate_grad_batches = 1
	batch_size = 16 // accumulate_grad_batches
	max_epochs = 610 * 2 # 1e6 / (13100 / 8)
	gn = 10
	n = 3
	iters_per_log = n * (10 // n)
	iters_per_sample = n * (500 // n)
	iters_per_ckpt = 10000

	################################
	# Model                        #
	################################
	up_scale = [2, 5, 2, 5, 2] # assert product = frame_shift
	sigma = 0.6
	n_flows = 4
	n_group = 8
	# for WN
	n_layers = 7
	n_channels = 128
	kernel_size = 3
	# for PF
	PF_n_layers = 7
	PF_n_channels = 64

	################################
	# Spectral Loss                #
	################################
	mag = True
	mel = True
	fft_sizes = [2048, 1024, 512, 256, 128]
	hop_sizes = [400, 200, 100, 50, 25]
	win_lengths = [2000, 1000, 500, 200, 100]
	mel_scales = [4, 2, 1, 0.5, 0.25]

lightning.seed_everything(hparams.seed)

def load_wav(path, seg = True):
	sr, wav = wavfile.read(path)
	wav = wav.astype(np.float32)
	wav = wav/np.max(np.abs(wav))
	try:
		assert sr == hparams.sample_rate
	except:
		print('Error:', path, 'has wrong sample rate.')
	if not seg:
		return wav
	if wav.shape[0] > hparams.seg_l:
		start = np.random.randint(0, len(wav)-hparams.seg_l)
		wav = wav[start:start+hparams.seg_l]
	else:
		wav = np.pad(wav, (0, hparams.seg_l-wav.shape[0]), 'constant', constant_values = (0, 0))
	return wav


def save_wav(wav, path):
	wav *= 32767 / max(0.01, np.max(np.abs(wav)))
	wavfile.write(path, hparams.sample_rate, wav.astype(np.int16))


def preemphasis(x):
	return scipy.signal.lfilter([1, -hparams.preemphasis], [1], x)


def inv_preemphasis(x):
	return scipy.signal.lfilter([1], [1, -hparams.preemphasis], x)


def spectrogram(y):
	D = _stft(preemphasis(y))
	S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
	return _normalize(S)


def inv_spectrogram(spectrogram):
	'''Converts spectrogram to waveform using librosa'''
	S = _db_to_amp(_denormalize(spectrogram) + hparams.ref_level_db)	# Convert back to linear
	return inv_preemphasis(_griffin_lim(S ** hparams.power))			# Reconstruct phase


def melspectrogram(y):
	D = _stft(preemphasis(y))
	S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
	return _normalize(S)


def inv_melspectrogram(spectrogram):
	mel = _db_to_amp(_denormalize(spectrogram) + hparams.ref_level_db)
	S = _mel_to_linear(mel)
	return inv_preemphasis(_griffin_lim(S ** hparams.power))


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
	window_length = int(hparams.sample_rate * min_silence_sec)
	hop_length = int(window_length / 4)
	threshold = _db_to_amp(threshold_db)
	for x in range(hop_length, len(wav) - window_length, hop_length):
		if np.max(wav[x:x+window_length]) < threshold:
			return x + hop_length
	return len(wav)


def _griffin_lim(S):
	'''librosa implementation of Griffin-Lim
	Based on https://github.com/librosa/librosa/issues/434
	'''
	angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
	S_complex = np.abs(S).astype(np.complex)
	y = _istft(S_complex * angles)
	for i in range(hparams.gl_iters):
		angles = np.exp(1j * np.angle(_stft(y)))
		y = _istft(S_complex * angles)
	return y


def _stft(y):
	n_fft, hop_length, win_length = _stft_parameters()
	return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):
	_, hop_length, win_length = _stft_parameters()
	return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_parameters():
	return (hparams.num_freq - 1) * 2, hparams.frame_shift, hparams.frame_length


# Conversions:

_mel_basis = None

def _linear_to_mel(spectrogram):
	global _mel_basis
	if _mel_basis is None:
		_mel_basis = _build_mel_basis()
	return np.dot(_mel_basis, spectrogram)
	

def _mel_to_linear(spectrogram):
	global _mel_basis
	if _mel_basis is None:
		_mel_basis = _build_mel_basis()
	inv_mel_basis = np.linalg.pinv(_mel_basis)
	inverse = np.dot(inv_mel_basis, spectrogram)
	inverse = np.maximum(1e-10, inverse)
	return inverse


def _build_mel_basis():
	n_fft = (hparams.num_freq - 1) * 2
	return librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=hparams.num_mels, fmin = hparams.fmin, fmax = hparams.fmax)


def _amp_to_db(x):
	return 20 * np.log10(np.maximum(1e-5, x))


def _db_to_amp(x):
	return np.power(10.0, x * 0.05)


def _normalize(S):
	return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)


def _denormalize(S):
	return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db

def files_to_list(fdir):
	f_list = []
	with open(os.path.join(fdir, 'metadata.csv'), encoding = 'utf-8') as f:
		for line in f:
			parts = line.strip().split('|')
			wav_path = os.path.join(fdir, 'wavs', '%s.wav' % parts[0])
			if hparams.prep:
				wav = load_wav(wav_path, False)
				if wav.shape[0] < hparams.seg_l:
					wav = np.pad(wav, (0, hparams.seg_l-wav.shape[0]), 'constant', constant_values = (0, 0))
				mel = melspectrogram(wav).astype(np.float32) 
				f_list.append([wav, mel])
			else:
				f_list.append(wav_path)
	if hparams.prep and hparams.pth is not None:
		with open(hparams.pth, 'wb') as w:
			pickle.dump(f_list, w)
	return f_list


class ljdataset(Dataset):
	def __init__(self, fdir):
		if hparams.prep and hparams.pth is not None and os.path.isfile(hparams.pth):
			with open(hparams.pth, 'rb') as r:
				self.f_list = pickle.load(r)
		else:
			self.f_list = files_to_list(fdir)

	def __getitem__(self, index):
		if hparams.prep:
			wav, mel = self.f_list[index]
			seg_ml = hparams.seg_l//hparams.frame_shift+1
			ms = np.random.randint(0, mel.shape[1]-seg_ml) if mel.shape[1] > seg_ml else 0
			ws = hparams.frame_shift*ms
			wav = wav[ws:ws+hparams.seg_l]
			mel = mel[:, ms:ms+seg_ml]
		else:
			wav = load_wav(self.f_list[index])
			mel = melspectrogram(wav).astype(np.float32)
		return wav, mel

	def __len__(self):
		return len(self.f_list)


def collate_fn(batch):
	wavs = []
	mels = []
	for wav, mel in batch:
		wavs.append(wav)
		mels.append(mel)
	wavs = torch.Tensor(wavs)
	mels = torch.Tensor(mels)
	return wavs, mels

def mode(obj, model = False):
	d = torch.device('cuda' if hparams.is_cuda else 'cpu')
	return obj.to(d, non_blocking = False if model else hparams.pin_mem)

def to_arr(var):
	return var.cpu().detach().numpy().astype(np.float32)

def prepare_dataloaders(fdir):
	trainset = ljdataset(fdir)
	train_loader = DataLoader(trainset, num_workers = hparams.n_workers, shuffle = True,
							  batch_size = hparams.batch_size, pin_memory = hparams.pin_mem,
							  drop_last = False, collate_fn = collate_fn)
	return train_loader

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
	n_channels_int = n_channels[0]
	in_act = input_a+input_b
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
		super(Invertible1x1Conv, self).__init__()
		self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
									bias=False); self.device = device

		# Sample a random orthonormal matrix to initialize weights
		W = torch.linalg.qr(torch.FloatTensor(c, c).normal_())[0]

		# Ensure determinant is 1.0 not -1.0
		if torch.det(W) < 0:
			W[:,0] = -1*W[:,0]
		W = W.view(c, c, 1)
		self.conv.weight.data = W

	def forward(self, z, reverse = False):
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
		W = self.conv.weight.squeeze()
		W_inverse = W.float().inverse()
		W_inverse = W_inverse[..., None]
		self.W_inverse = W_inverse
		self.set = True
    
class WN(torch.nn.Module):
	def __init__(self, n_in_channels, n_emb_channels, PF = False):
		super(WN, self).__init__()
		assert(hparams.kernel_size % 2 == 1)
		assert(hparams.n_channels % 2 == 0)
		self.PF = PF
		self.n_layers = hparams.PF_n_layers if PF else hparams.n_layers
		self.n_channels = hparams.PF_n_channels if PF else hparams.n_channels
		self.in_layers = torch.nn.ModuleList()
		self.cond_layers = torch.nn.ModuleList()
		self.res_skip_layers = torch.nn.ModuleList()
#         self.register_buffer("n_channels_tensor", torch.IntTensor([self.n_channels]))
        
		start = torch.nn.Conv1d(n_in_channels, self.n_channels, 1)
		start = torch.nn.utils.weight_norm(start, name = 'weight')
		self.start = start

		if PF:
			self.end = torch.nn.Conv1d(self.n_channels, n_in_channels, 1)
		else:
			self.end = torch.nn.Conv1d(self.n_channels, 2*n_in_channels, 1)
			self.end.weight.data.zero_()
			self.end.bias.data.zero_()

		for i in range(self.n_layers):
			dilation = 2**i if PF else 2**i
			padding = int((hparams.kernel_size*dilation - dilation)/2)
			in_layer = torch.nn.Conv1d(self.n_channels, 2*self.n_channels, hparams.kernel_size,
									   dilation = dilation, padding = padding)
			in_layer = torch.nn.utils.weight_norm(in_layer, name = 'weight')
			self.in_layers.append(in_layer)

			cond_layer = torch.nn.Conv1d(n_emb_channels, 2*self.n_channels, 1)
			cond_layer = torch.nn.utils.weight_norm(cond_layer, name = 'weight')
			self.cond_layers.append(cond_layer)

			# last one is not necessary
			if i < self.n_layers - 1:
				res_skip_channels = 2*self.n_channels
			else:
				res_skip_channels = self.n_channels
			res_skip_layer = torch.nn.Conv1d(self.n_channels, res_skip_channels, 1)
			res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name = 'weight')
			self.res_skip_layers.append(res_skip_layer)

	def forward(self, wavs, mels):
# 		print(f"start: {self.start.weight.dtype}");
# 		print(f"wavs: {wavs.dtype}");
		wavs = self.start(wavs)

		for i in range(self.n_layers):
			acts = fused_add_tanh_sigmoid_multiply(
				self.in_layers[i](wavs),
				self.cond_layers[i](mels),
				torch.IntTensor([self.n_channels]))

			res_skip_acts = self.res_skip_layers[i](acts)
			if i < self.n_layers - 1:
				wavs = res_skip_acts[:,:self.n_channels,:] + wavs
				skip_acts = res_skip_acts[:,self.n_channels:,:]
			else:
				skip_acts = res_skip_acts

			if i == 0:
				output = skip_acts
			else:
				output = skip_acts + output
		return self.end(torch.nn.ReLU()(output) if self.PF else output)

class US(torch.nn.Module):
	def __init__(self):
		super(US, self).__init__()
		self.layers = torch.nn.ModuleList()
		self.layers.append(torch.nn.Conv1d(hparams.num_mels, hparams.num_mels, kernel_size = 2))
		self.layers.append(torch.nn.ReLU())
		for sf in hparams.up_scale:		
			self.layers.append(torch.nn.Upsample(scale_factor = sf))
			self.layers.append(torch.nn.Conv1d(hparams.num_mels, hparams.num_mels,
											kernel_size = 5, padding = 2))
			self.layers.append(torch.nn.ReLU())

	def forward(self, mels):
		for f in self.layers:
			mels = f(mels)
		return mels

class Model(torch.nn.Module):
	def __init__(self, device):
		super(Model, self).__init__()
		assert(hparams.n_group % 2 == 0)
		self.n_group = hparams.n_group
		self.n_flows = hparams.n_flows
		self.upsample = US()
		self.WN = WN(int(hparams.n_group/2), hparams.num_mels*hparams.n_group)
		self.convinv = torch.nn.ModuleList()
		self.device = device
		for k in range(hparams.n_flows):
			self.convinv.append(Invertible1x1Conv(hparams.n_group, device))
		self.PF = WN(1, hparams.num_mels, True)
# 		print(f"Model device: {device}")

	def forward(self, wavs, mels):
		'''
		wavs: (batch_size, seg_l)
		mels: (batch_size, num_mels, T)
		'''
		#  Upsample spectrogram to size of audio
		mels = self.upsample(mels)
		assert(mels.size(2) == wavs.size(1))

		mels = mels.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3) # (batch_size, seg_l//n_group, num_mels, n_group)
		mels = mels.contiguous().view(mels.size(0), mels.size(1), -1).permute(0, 2, 1) # (batch_size, num_mels*n_group, seg_l//n_group)

		audio = wavs.unfold(1, self.n_group, self.n_group).permute(0, 2, 1) # (batch_size, n_group, seg_l//n_group)

		log_s_list = []
		log_det_W_list = []

# 		print(f'n_flows: {self.n_flows}')
		for k in range(self.n_flows):
			audio, log_det_W = self.convinv[k](audio)
# 			print(f'k: {k}')
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
		'''
		mels: (batch_size, num_mels, T)
		'''
		# (batch_size, T//n_group, num_mels, n_group)
		mels = inp_mels.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
		# (batch_size, num_mels*n_group, T//n_group)
		mels = mels.contiguous().view(mels.size(0), mels.size(1), -1).permute(0, 2, 1)

		audio = torch.FloatTensor(mels.size(0),
								hparams.n_group,
								mels.size(2)).normal_().cuda()
# 		print(f"audio: {audio.device}")

		# audio = mode(hparams.sigma*audio)
		audio = hparams.sigma * audio

		for k in reversed(range(self.n_flows)):
			n_half = int(audio.size(1)/2)
			audio_0 = audio[:,:n_half,:]
			audio_1 = audio[:,n_half:,:]

			output = self.WN(audio_0, mels)
			s = output[:, n_half:, :]
			b = output[:, :n_half, :]
			audio_1 = (audio_1 - b)/torch.exp(s)
			audio = torch.cat([audio_0, audio_1],1)

			audio = self.convinv[k](audio, reverse = True)

		audio = audio.permute(0, 2, 1).contiguous().view(audio.size(0), 1, -1) # (batch_size, 1, seg_l)
		return audio

	def infer(self, mels):
		'''
		mels: (batch_size, num_mels, T')
		'''
		inp_mels = self.upsample(mels) # (batch_size, num_mels, T)
# 		print(f'inp_mels: {inp_mels.device}')
		audio = self.WG(inp_mels)
# 		print(f"audio: {audio.shape}")
		d = inp_mels.size(2)-audio.size(2)
		if d > 0:
			audio = torch.cat([audio, 0*audio[:, :, :d]], 2)
		audio = self.PF(audio, inp_mels).squeeze(1)
		return audio

	def set_inverse(self):
		for i in range(hparams.n_flows):
			self.convinv[i].set_inverse()
	
	@staticmethod
	def remove_weightnorm(model):
		waveglow = model
		for WN in [waveglow.WN, waveglow.PF]:
			WN.start = torch.nn.utils.remove_weight_norm(WN.start)
			WN.in_layers = remove(WN.in_layers)
			WN.cond_layers = remove(WN.cond_layers)
			WN.res_skip_layers = remove(WN.res_skip_layers)
		return waveglow

def remove(conv_list):
	new_conv_list = torch.nn.ModuleList()
	for old_conv in conv_list:
		old_conv = torch.nn.utils.remove_weight_norm(old_conv)
		new_conv_list.append(old_conv)
	return new_conv_list


class Loss(torch.nn.Module):
	def __init__(self, device):
		super(Loss, self).__init__()
		self.d = 2*hparams.sigma*hparams.sigma
		self.loss = MultiResolutionSTFTLoss(device, hparams.fft_sizes, hparams.hop_sizes,
											hparams.win_lengths, hparams.mel_scales)
# 		print(f"Loss device: {device}")

	def forward(self, model_output, p_wavs = None, r_wavs = None):
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
	# ref: https://github.com/kan-bayashi/ParallelWaveGAN
	"""Multi resolution STFT loss module."""
	def __init__(self, device,
				 fft_sizes=[1024, 2048, 512],
				 hop_sizes=[120, 240, 50],
				 win_lengths=[600, 1200, 240],
				 mel_scales=[1, 1, 1],
				 window="hann_window"):
		"""Initialize Multi resolution STFT loss module.

		Args:
			fft_sizes (list): List of FFT sizes.
			hop_sizes (list): List of hop sizes.
			win_lengths (list): List of window lengths.
			window (str): Window function type.

		"""
		super(MultiResolutionSTFTLoss, self).__init__()
		assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
		self.stft_losses = torch.nn.ModuleList()
		self.bases = []
		for fs, ss, wl, sc in zip(fft_sizes, hop_sizes, win_lengths, mel_scales):
			self.stft_losses += [STFTLoss(device, fs, ss, wl, window)]
			b = librosa.filters.mel(hparams.sample_rate, fs, n_mels = hparams.num_mels*sc, fmax = hparams.fmax).T
			# self.bases += [mode(torch.Tensor(b))]
			self.bases += [torch.Tensor(b, device=device)]

	def forward(self, x, y):
		"""Calculate forward propagation.

		Args:
			x (Tensor): Predicted signal (B, T).
			y (Tensor): Groundtruth signal (B, T).

		Returns:
			Tensor: Multi resolution spectral convergence loss value.
			Tensor: Multi resolution log spectral loss value.

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
	"""STFT loss module."""

	def __init__(self, device, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
		"""Initialize STFT loss module."""
		super(STFTLoss, self).__init__()
		self.fft_size = fft_size
		self.shift_size = shift_size
		self.win_length = win_length
		# self.window = mode(getattr(torch, window)(win_length))
		self.window = getattr(torch, window)(win_length, device=device).cuda()
# 		print(f"window device: {device}")
# 		print(f"window: {self.window.device}")

	def forward(self, x, y, b):
		"""Calculate forward propagation.

		Args:
			x (Tensor): Predicted signal (B, T).
			y (Tensor): Groundtruth signal (B, T).
			b (Tensor): Mel basis (fft_size//2+1, num_mels).

		Returns:
			Tensor: Spectral convergence loss value.
			Tensor: Log STFT magnitude loss value.

		"""
		x_mag, x_mel = stft(x, self.fft_size, self.shift_size, self.win_length, self.window, b)
		y_mag, y_mel = stft(y, self.fft_size, self.shift_size, self.win_length, self.window, b)
		sc_loss = spec_loss = 0
		if hparams.mag:
			h = x_mag.size(2)*2*hparams.fmax//hparams.sample_rate if hparams.sample_rate >= 2*hparams.fmax else x_mag.size(2)
			x_mag_ = x_mag[:, :, :h]
			y_mag_ = y_mag[:, :, :h]
			sc_loss += torch.norm((y_mag_-x_mag_), p = "fro")/torch.norm(y_mag_, p = "fro")
			spec_loss += torch.nn.L1Loss()(torch.log(x_mag_), torch.log(y_mag_))
			if h < x_mag.size(2):
				x_mag_m = x_mag[:, :, h:].mean(1)
				y_mag_m = y_mag[:, :, h:].mean(1)
				sc_loss += torch.norm((y_mag_m-x_mag_m), p = "fro")/torch.norm(y_mag_m, p = "fro")
				spec_loss += torch.nn.L1Loss()(torch.log(x_mag_m), torch.log(y_mag_m))
		if hparams.mel:
			sc_loss += torch.norm((y_mel-x_mel), p = "fro")/torch.norm(y_mel, p = "fro")
			spec_loss += torch.nn.L1Loss()(torch.log(x_mel), torch.log(y_mel))
		s = int(hparams.mag)+int(hparams.mel)
		if s == 0:
			print('Error: hparams.mag and hparams.mel are both set as False.')
			exit()
		return sc_loss/s, spec_loss/s


def stft(x, fft_size, hop_size, win_length, window, b):
	"""Perform STFT and convert to magnitude spectrogram.

	Args:
		x (Tensor): Input signal tensor (B, T).
		fft_size (int): FFT size.
		hop_size (int): Hop size.
		win_length (int): Window length.
		window (str): Window function type.
		b (Tensor): Mel basis (fft_size//2+1, num_mels).

	Returns:
		Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

	"""
	x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
	real = x_stft[..., 0]
	imag = x_stft[..., 1]

	# NOTE(kan-bayashi): clamp is needed to avoid nan or inf
	mag = torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)
	return mag, torch.clamp(torch.matmul(mag.cuda(), b.cuda()), min = 1e-7**0.5)

train_dataloader = prepare_dataloaders(hparams.train_dir)
print(f"Total batches: {len(train_dataloader)}")

current_epoch = 309

class WGWaveNetAlgorithm(lightning.LightningModule):

    def __init__(self):
        super().__init__()
        # print(f"Algo device: {self.device}")
        self.model = Model(self.device)
        self.criterion = Loss(self.device)

    def forward(self, batch):
        return batch

    def training_step(self, batch, batch_idx):

        # Get data samples.
        wavs, mels = batch

        # Forward pass.
        outputs = self.model(wavs, mels)
        # print(f"outputa: {len(outputs)}")
        p_wavs = self.model.infer(mels) if batch_idx % hparams.n == 0 else None
        # print(f"p_wavs: {p_wavs != None}")

        # Compute loss.
        total_loss, zloss, sloss = self.criterion(outputs, p_wavs, wavs)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = hparams.lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, hparams.sch_step, hparams.sch_g, last_epoch=-1)
        
        # Update the learning rate according to scheduler.
        for _ in range(current_epoch):
            lr_scheduler.step()
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }

if __name__ == '__main__':

    # WG-WaveNet Algorithm
    wgwavenet_algorithm = WGWaveNetAlgorithm()

    # Others
    training_session = 5
    ckpt_path = f"../input/wg-wavenet-lightning-v2/lightning_logs/version_0/checkpoints/epoch={current_epoch}-step=253384.ckpt"

    # Trainer
    trainer = lightning.Trainer(
        max_time={'hours': 11 * training_session, 'minutes': 50 * training_session},#"00:08:50:00",
        max_epochs=hparams.max_epochs,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        precision=16,
    #     gpus=-1,
        accelerator="auto",
        devices=-1,
        # tpu_cores=8,
        reload_dataloaders_every_n_epochs=1,
    )

    # Fit the model to dataset.
    trainer.fit(
        wgwavenet_algorithm, 
        train_dataloader,
        ckpt_path=ckpt_path
    )