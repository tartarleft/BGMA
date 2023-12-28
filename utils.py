import random
import librosa
import numpy as np
import torch
import scipy
import torch.nn.init as init
import math
import os
import yaml
from torch.optim import lr_scheduler

seed = 8
random.seed(seed)
np.random.seed(seed)


class Config():

    def __init__(self):
        self.ebds_path = './data/TIMIT/spkEBDs'
        self.data_path = './data/TIMIT/test_wavs'
        self.preTrain_path = './ckpt/TIMIT'
        self.modeltype = 'ECAPA'
        self.n_mels = 40
        self.freq_range = 257
        self.batch_length = 16000 * 3
        self.mel_filter = librosa.filters.mel(sr=16000, n_fft=512, n_mels=self.n_mels)
        self.mel_list = self.mel_filter.argmax(axis=1)
        self.mel_empty_list = [x for x in range(self.freq_range) if x not in self.mel_list]
        self.initialize = 'scale'
        self.music_nmel = 256
        self.music_n_fft = 2048
        self.music_mel_filter = librosa.filters.mel(sr=16000, n_fft=self.music_n_fft, n_mels=self.music_nmel)



    def set_dataset(self, dataset):
        self.dataset = dataset
        if dataset == 'TIMIT':
            self.ebds_path = './data/TIMIT/spkEBDs'
            self.data_path = './data/TIMIT/test_wavs'
            self.preTrain_path = './ckpt/TIMIT'
        else:
            print('Dataset Error')
            input()


# config = Config()

# def getMel(utter):
# 	S = librosa.core.stft(y=utter, n_fft=512, win_length=400, hop_length=160)
# 	S = np.clip(S, -50, 50)
# 	S = np.abs(S) ** 2
# 	mel_basis = librosa.filters.mel(sr=16000, n_fft=512, n_mels=40)
# 	S = np.log10(np.dot(mel_basis, S) + 1e-6)
# 	# return S[:, :300]
# 	return S


def getMel(utter):
    S = torch.stft(utter, 512, hop_length=160, win_length=400, window=torch.hann_window(400).cuda(), return_complex=False)
    # S = torch.clip(S, -50, 50)
    # S = torch.abs(S)**2
    S = S.pow(2).sum(-1)
    mel_basis = librosa.filters.mel(sr=16000, n_fft=512, n_mels=40)
    mel_basis = torch.from_numpy(mel_basis).cuda()
    S = torch.log10(torch.mm(mel_basis, S) + 1e-6)
    return S


def getSpect(utter):
    spect = torch.stft(torch.tensor(utter), 512, hop_length=160, win_length=400, window=torch.hann_window(400))
    # spect = spect.clip(-50, 50)
    spect = spect.pow(2).sum(-1)
    return spect.numpy()


def getMelnoLog(utter):
    S = librosa.core.stft(y=utter, n_fft=512, win_length=400, hop_length=160)
    # S = np.clip(S, -50, 50)
    S = np.abs(S)**2
    mel_basis = librosa.filters.mel(sr=16000, n_fft=512, n_mels=40)
    S = np.dot(mel_basis, S)
    # return S[:, :300]
    return S


def CosineSimilarity(a, b):
    a = torch.tensor(a)
    b = torch.tensor(b)
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    simmatrix = torch.nn.functional.cosine_similarity(a, b, dim=1).cpu().detach().numpy()
    return simmatrix


def CosineSimilarityGC(a, b):
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    simmatrix = torch.nn.functional.cosine_similarity(a, b, dim=1)
    return simmatrix


def EditDistance(a, b):
    matrix = [[i + j for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j - 1] + d, matrix[i - 1][j] + 1, matrix[i][j - 1] + 1)
    return matrix[-1][-1] / len(a)


def ReLU(x):
    return np.maximum(x, 0.0)


def chk_NaN(x):
    flag = False
    if np.isinf(x).any():
        print('inf error')
        flag = True
    if np.isnan(x).any():
        print('nan error')
        flag = True
    if flag is True:
        print('*' * 10)


def get_spectrogram(data, config, win_len=None):
    ### get spectrogram according to the configuration and window_length
    ### we first calculate the power2-spectrum,
    ### and then get the Mel-spectrogram via the Mel-Filter banks
    stft_matrix = librosa.stft(data, n_fft=config['n_fft'], hop_length=config['hop_length'], win_length=win_len)
    mag_D = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)
    pwr = mag_D**2
    mel_basis = librosa.filters.mel(sr=config['sr'], n_fft=config['n_fft'], n_mels=config['n_mels'])
    mel_pwr = np.dot(mel_basis, pwr)
    chk_NaN(mel_pwr)
    # last, apply the gamma-power to approxiate Steven power law.
    return mel_pwr**0.3, phase, pwr


def get_cepstrogram(spec, config):
    ### return the cepstrogram according to the spectrogram
    ### spec.shape = [256,302]                 
    mel_ceps_coef = scipy.fftpack.dct(spec, axis=0, type=config['dct_type'], norm=config['norm'])
    mel_ceps_coef_relu = np.maximum(mel_ceps_coef, 0.0)
    chk_NaN(mel_ceps_coef_relu)
    return mel_ceps_coef_relu


def get_diff_spectrogram(spec, config):
    # only to diff by time
    mode = config['d_spec_type']
    d_spec = np.zeros_like(spec)
    hei, wid = d_spec.shape
    for i in range(1, wid - 1):
        if mode == 'all':  # nxt - pre
            d_spec[:, i] = spec[:, i + 1] - spec[:, i - 1]
        elif mode == 'decay':  # ReLU(-all)
            d_spec[:, i] = ReLU(spec[:, i - 1] - spec[:, i + 1])
        elif mode == 'attack':  # ReLU(all)
            d_spec[:, i] = ReLU(spec[:, i + 1] - spec[:, i - 1])
    d_spec[:, 0] = d_spec[:, 1]
    d_spec[:, -1] = d_spec[:, -2]
    return d_spec


def get_spectral_envelope(mel_spec, config):
    MFCC = scipy.fftpack.dct(mel_spec, axis=0, type=config['dct_type'], norm=config['norm'])
    hei, wid = MFCC.shape
    MFCC[15:, :] = 0.0
    ret = scipy.fftpack.idct(MFCC, axis=0, type=config['dct_type'], norm=config['norm'])
    return ReLU(ret)


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [
        os.path.join(dirname, f) for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f
    ]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'],
                                        last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):

    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)
    
    
def spsi(msgram, fftsize, hop_length):
    """ https://github.com/lonce/SPSI_Python
    Takes a 2D spectrogram ([freqs,frames]), the fft legnth (= widnow length) and the hope size (both in units of samples).
    Returns an audio signal.
    """

    numBins, numFrames = msgram.shape
    y_out = np.zeros(numFrames * hop_length + fftsize - hop_length)

    m_phase = np.zeros(numBins)
    m_win = scipy.signal.hanning(fftsize,
                                 sym=True)  # assumption here that hann was used to create the frames of the spectrogram

    #processes one frame of audio at a time
    for i in range(numFrames):
        m_mag = msgram[:, i]
        for j in range(1, numBins - 1):
            if (m_mag[j] > m_mag[j - 1] and m_mag[j] > m_mag[j + 1]):  #if j is a peak
                alpha = m_mag[j - 1]
                beta = m_mag[j]
                gamma = m_mag[j + 1]
                denom = alpha - 2 * beta + gamma

                if (denom != 0):
                    p = 0.5 * (alpha - gamma) / denom
                else:
                    p = 0

                phaseRate = 2 * np.pi * (j + p) / fftsize
                #adjusted phase rate
                m_phase[j] = m_phase[j] + hop_length * phaseRate
                #phase accumulator for this peak bin
                peakPhase = m_phase[j]

                # If actual peak is to the right of the bin freq
                if (p > 0):
                    # First bin to right has pi shift
                    bin = j + 1
                    m_phase[bin] = peakPhase + np.pi

                    # Bins to left have shift of pi
                    bin = j - 1
                    while ((bin > 1) and (m_mag[bin] < m_mag[bin + 1])):  # until you reach the trough
                        m_phase[bin] = peakPhase + np.pi
                        bin = bin - 1

                    #Bins to the right (beyond the first) have 0 shift
                    bin = j + 2
                    while ((bin < (numBins)) and (m_mag[bin] < m_mag[bin - 1])):
                        m_phase[bin] = peakPhase
                        bin = bin + 1

                #if actual peak is to the left of the bin frequency
                if (p < 0):
                    # First bin to left has pi shift
                    bin = j - 1
                    m_phase[bin] = peakPhase + np.pi

                    # and bins to the right of me - here I am stuck in the middle with you
                    bin = j + 1
                    while ((bin < (numBins)) and (m_mag[bin] < m_mag[bin - 1])):
                        m_phase[bin] = peakPhase + np.pi
                        bin = bin + 1

                    # and further to the left have zero shift
                    bin = j - 2
                    while ((bin > 1) and (m_mag[bin] < m_mag[bin + 1])):  # until trough
                        m_phase[bin] = peakPhase
                        bin = bin - 1

            #end ops for peaks
        #end loop over fft bins with

        magphase = m_mag * np.exp(1j * m_phase)  #reconstruct with new phase (elementwise mult)
        magphase[0] = 0
        magphase[numBins - 1] = 0  #remove dc and nyquist
        m_recon = np.concatenate([magphase, np.flip(np.conjugate(magphase[1:numBins - 1]), 0)])

        #overlap and add
        m_recon = np.real(np.fft.ifft(m_recon)) * m_win
        y_out[i * hop_length:i * hop_length + fftsize] += m_recon
    return y_out



def spsi_eff(magD, y_out, fftsize, hop_length):
    p = np.angle(librosa.stft(y_out, fftsize, hop_length, center=False))
    for i in range(50):
        S = magD * np.exp(1j * p)
        x = librosa.istft(S, hop_length, win_length=fftsize,
                          center=True)  # Griffin Lim, assumes hann window; librosa only does one iteration?
        p = np.angle(librosa.stft(x, fftsize, hop_length, center=True))
    return p