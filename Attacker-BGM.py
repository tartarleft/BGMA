from __future__ import generator_stop
import os
import pickle
import argparse
import librosa
from trainer import MUNIT_Trainer
import numpy as np
import torch
import soundfile as sf
from torch.autograd import grad
from SRS import SRS
from utils import *
from tqdm import tqdm
from musicPreprocess import get_musics
import tensorboardX


class Attacker():
    def __init__(self, srs, config=None):
        self.srs = srs
        self.data = self.__loadFullData()
        self.spknum = len(self.data)
        assert (self.spknum == self.srs.spknum)
        self.spklist = self.srs.getSpkList()
        self.advspklist = self.spklist.copy()
        for idx in range(len(self.advspklist)):
            self.advspklist[idx] = 'adv_' + \
                str(idx) + '_' + self.advspklist[idx]
        self.musics = get_musics()
        self.music_config = {
            'sr': 16000,
            'n_fft': 2048,
            'hop_length': 160,
            'n_mels': 256,
            'dct_type': 2,
            'norm': 'ortho',
            'd_spec_type': 'attack',
        }
        self.threshdict = {'TDNN': 0.29556171,
                           'Dense': 0.29594862, 'ECAPA': 0.27176976}

    def __loadFullData(self):
        data = {}
        spkwavs = os.listdir(config.data_path)
        print('loading data from path %s' % config.data_path)
        for spk in spkwavs:
            with open(os.path.join(config.data_path, spk), 'rb') as f:
                utterlist = pickle.load(f)
            data[spk] = utterlist
        return data

    def __getMaxUtterLen(self):
        maxlen = -99
        for spk in self.spklist:
            for utter in self.data[spk]:
                ulen = len(utter)
                if ulen > maxlen:
                    maxlen = ulen
        return maxlen

    def clipMusicSlice(self, maxlen=-1, mode='f'):
        print('music clip mode: ', mode)
        if maxlen < 0:
            maxlen = self.__getMaxUtterLen()
        for i in range(len(self.musics)):
            if mode == 'f':
                self.musics[i] = self.musics[i][:maxlen]
            elif mode == 'b':
                self.musics[i] = self.musics[i][-maxlen:]
            elif mode == 'r':
                music_start = np.random.randint(len(self.musics[i]) - maxlen)
                self.musics[i] = self.musics[i][music_start:music_start + maxlen]
        return

    @staticmethod
    def OHE(y, bsize, csize):
        one_hot = torch.FloatTensor(bsize, csize).cuda()
        one_hot.zero_()
        one_hot.scatter_(1, y.view(bsize, 1), 1)
        return one_hot

    def get_mel_inv(self, filter=None):
        if filter is None:
            filter = librosa.filters.mel(sr=16000, n_fft=2048, n_mels=256)
        U, s, V = np.linalg.svd(filter)
        us, vs = U.shape[0], V.shape[0]
        assert vs > us, 'row larger than column, change method'
        V21 = V[us:vs, 0:us]
        V22 = V[us:vs, us:vs]
        VT12 = V.T[0:us, us:vs]
        VT22 = V.T[us:vs, us:vs]
        Cmat = np.block([[VT12 @ V21, VT12 @ V22], [VT22 @ V21, VT22 @ V22]])
        filter_inv = np.linalg.pinv(filter)
        # matS = np.random.rand(1025, 500)
        # m = filter @ matS
        # print('inv matrix test:', np.allclose(matS, filter_inv @ m + Cmat @ matS))
        # input()
        self.filter = torch.from_numpy(filter).cuda()
        self.filter_inv = torch.from_numpy(filter_inv).cuda()
        self.Cmat = torch.from_numpy(Cmat).cuda()
        return self.filter_inv, self.Cmat

    def wavRecon(self, mel_pwr, phase, mag, mlen=0):
        mag_recon = torch.mm(self.filter_inv, mel_pwr) + \
            torch.mm(self.Cmat, mag)
        mag_recon = torch.relu(mag_recon)
        real = mag_recon**0.5 * torch.cos(phase)
        img = mag_recon**0.5 * torch.sin(phase)
        wav = torch.istft(torch.stack((real, img), dim=2), 2048, 160)
        # scale_rate = torch.abs(wav).max()
        # wav = wav / scale_rate
        return wav

    def get_encoder_feature(self, music):
        mel_pwr, phase, mag = get_spectrogram(music, self.music_config)
        ceps = get_cepstrogram(mel_pwr, self.music_config)
        d_spec = get_diff_spectrogram(mel_pwr, self.music_config)
        enve = get_spectral_envelope(mel_pwr, self.music_config)
        input_feature = np.stack((mel_pwr, ceps, d_spec, enve), axis=0)
        return torch.from_numpy(input_feature).cuda(), torch.from_numpy(phase).cuda(), torch.from_numpy(mag).cuda()


# Max Embedding Attacking Loss


    def MaxEmdLoss(self, sim, sidx):
        sidx = torch.LongTensor([sidx]).cuda()
        lable_mask = self.OHE(sidx, sidx.size(0), self.spknum)
        correct_sim = torch.sum(lable_mask * sim, 1)
        wrong_sim = torch.max((1 - lable_mask) * sim - 1e4 * lable_mask, 1)[0]
        loss = -torch.nn.ReLU()(correct_sim - wrong_sim + 0.3)
        return torch.mean(loss)

    def TgtEmdLoss(self, sim, sidx, tidx):
        sidx = torch.LongTensor([sidx]).cuda()
        tidx = torch.LongTensor([tidx]).cuda()
        lable_mask = self.OHE(sidx, sidx.size(0), self.spknum)
        correct_sim = torch.sum(lable_mask * sim, 1)
        tgt_mask = self.OHE(tidx, tidx.size(0), self.spknum)
        tgt_sim = torch.sum(tgt_mask * sim, 1)
        loss = -torch.nn.ReLU()(correct_sim - tgt_sim + 0.3)
        return torch.mean(loss)

    def MusicTgtLoss(self, sim, tidx):
        tidx = torch.LongTensor([tidx]).cuda()
        tgt_mask = self.OHE(tidx, tidx.size(0), self.spknum)
        tgt_sim = torch.sum(tgt_mask * sim, 1)
        # omax_sim = torch.max((1 - tgt_mask) * sim - 1e4 * tgt_mask, 1)[0]
        loss = -torch.nn.ReLU()(-tgt_sim + 1)
        return torch.mean(loss)

    def Projection(self, x, x_nat, epsilon):
        diff = x - x_nat
        diff = torch.clamp(diff, -epsilon, epsilon)
        x = x_nat + diff
        x = torch.clamp(x, -1, 1)
        return x

    def ProjectionL2(self, x, x_nat, epsilon):
        diff = x - x_nat
        diff = diff.view(1, -1).renorm(p=2, dim=0, maxnorm=epsilon)
        x = x_nat + diff.squeeze()
        x = torch.clamp(x, -1, 1)
        return x

# PGD Attacks

    def MusicPGD(self, model, utter, epsilon, iter, beta=0, label=-1, mode='linf', early_stop=False):
        utter = torch.from_numpy(utter).cuda()
        uttero = utter.clone().detach()
        if mode == 'l2':
            step_size = epsilon / 4
        else:
            step_size = 0.002
        utter.requires_grad = True
        v = 0
        threshold = self.threshdict[self.srs.modeltype]
        step = iter
        for i in range(iter):
            simM = model.QuerySimGC(utter)
            if early_stop:
                pred_adv = torch.argmax(simM)
                score = simM[pred_adv]
                if pred_adv == label and score > threshold:
                    step = i + 1
                    return utter.detach(), step
            loss = self.MusicTgtLoss(simM, label)
            ugrad = grad(loss, utter)[0]
            v = beta * v + ugrad
            if mode == 'linf':
                utter = utter + step_size * torch.sign(v)
                utter = self.Projection(utter, uttero, epsilon)
            elif mode == 'l2':
                utter = utter + step_size * (v / (torch.norm(v) + 1e-10))
                utter = self.ProjectionL2(utter, uttero, epsilon)
            else:
                assert False, "Wrong constraint mode"
            model.net.net.zero_grad()

        if early_stop:
            return utter.detach(), iter

        return utter.detach()

# BGM Attack

    def BGMA(self, music, encoder, decoder, iter, logger, beta=0, label=-1, index=''):
        spec_features, phase, mag = self.get_encoder_feature(music)
        loss_list, score_list = [], []
        threshold = self.threshdict[self.srs.modeltype]
        with torch.no_grad():
            content, style = encoder.encode(spec_features.unsqueeze(0))
        content_adv = content.detach()
        content_adv.requires_grad = True
        v, loss = 0, 0
        suc = False
        lr = 0.01
        step = iter
        with tqdm(total=iter) as _tqdm:
            for i in range(iter):
                _tqdm.set_description('iter: {}/{}'.format(i, iter))
                mel_recon = decoder.decode(content_adv, style)[0][0]
                wave_recon = self.wavRecon(mel_recon**(1 / 0.3), phase, mag)
                simM = self.srs.QuerySimGC(wave_recon)
                predic = np.argmax(simM.detach().cpu().numpy())
                score = simM.detach().cpu().numpy()[label]
                score_list.append(score)
                if predic == label and score > threshold:
                    step = i + 1
                    suc = True
                    break
                loss = self.MusicTgtLoss(simM, label)
                ugrad = grad(loss, content_adv)[0]
                v = beta * v + ugrad
                content_adv = content_adv + lr * v
                with torch.no_grad():
                    mag = torch.stft(wave_recon.detach(),
                                     n_fft=2048, hop_length=160)
                    mag = mag.pow(2).sum(-1)
                loss_list.append(loss.detach().cpu().numpy())
                _tqdm.set_postfix(loss='{:.6f}'.format(loss_list[-1]))
                logger.add_scalar('loss' + str(index),
                                  loss.detach().cpu().numpy(), i + 1)
                logger.add_scalar('score' + str(index), score, i + 1)
                _tqdm.update(1)
                decoder.zero_grad()
                self.srs.net.net.zero_grad()

        return wave_recon, suc, loss_list, score_list, step


# Experiments


    def MusicAttackDirectRandtgt(self, epsilon, iter, beta, mode='linf', model=None):
        print('Start Testing Model {} for direct prediction results'.format(config.modeltype))
        if model is None:
            model = self.srs
        direc_path = os.path.join('./results/PGD', self.srs.modeltype)
        if not os.path.exists(direc_path):
            os.makedirs(direc_path)
        exp_name = str(epsilon) + mode + '_' + str(iter) + "_" + str(beta)
        log_path = os.path.join(direc_path, exp_name)
        wave_path = os.path.join(direc_path, exp_name + '_wav')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if not os.path.exists(wave_path):
            os.makedirs(wave_path)
        threshold = self.threshdict[self.srs.modeltype]
        music_num = len(self.musics)

        adv_idx, scores, steps = [], [], []
        asr = 0
        for idx, music in tqdm(enumerate(self.musics), total=music_num):
            rand_spk = np.random.choice(np.arange(63), 20)
            for tgt_spk in rand_spk:
                # print('Target Speaker is %d' % tgt_spk)
                adv_music, step = self.MusicPGD(model, music, epsilon, iter, beta, tgt_spk, mode, early_stop=True)
                with torch.no_grad():
                    pred_adv = self.srs.Query(adv_music)
                    adv_idx.append(pred_adv)
                    simM = self.srs.QuerySimGC(adv_music)
                    score = simM.detach().cpu().numpy()[tgt_spk]
                    scores.append(score)
                    if pred_adv == tgt_spk and score > threshold:
                        asr += 1
                steps.append(step)
            adv_save = adv_music.detach().cpu().numpy()
            wave_name = 'adv_' + str(idx) + '.wav'
            sf.write(os.path.join(wave_path, wave_name), adv_save, samplerate=16000)
        asr = asr / len(self.musics * 20) * 100
        print('asr : %f' % (asr))
        print('avg score : %f' % np.mean(scores))
        print('avg step : %f' % np.mean(steps))
        return asr, np.mean(scores), steps

    def BGMARandTgt(self, iter, beta):
        print('Start Testing Model {} for style attack'.format(self.srs.modeltype))
        print('Total Speaker num %d' % self.spknum)
        attack_writer = tensorboardX.SummaryWriter('./attack_log')
        
        # initiliaze the inverse matrix for mel-filter bank
        self.get_mel_inv()
        
        # load music encoder and decoder
        generator_path = './ckpt/gen_00600000.pt'
        generator_config = get_config('./config/pia2gui_nfft2048_mel256.yaml')
        trainer = MUNIT_Trainer(generator_config)
        state_dict = torch.load(generator_path)
        trainer.gen_a.load_state_dict(state_dict['a'])
        trainer.gen_b.load_state_dict(state_dict['b'])
        trainer.cuda()
        trainer.eval()
        log_path = os.path.join('./results/BGMA/', self.srs.modeltype)
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        # sf.write(os.path.join(log_path, 'original_music.wav'), music, samplerate=16000)
        scores, asr, steps = [], [], []
        num_music = len(self.musics)

        for i, music in enumerate(self.musics):
            print('music %d of %d' % (i+1, num_music))
            rand_spk = np.random.choice(np.arange(63), 20)
            for tgt_spk in rand_spk:
                adv_music, suc, _, score_list, step = self.BGMA(music, trainer.gen_a, trainer.gen_a, iter, attack_writer,
                                                                beta, tgt_spk, i)
                music_path = os.path.join(log_path, 'adv_music_{}_spk{}.wav'.format(str(i), str(tgt_spk)))
                sf.write(music_path, adv_music.detach().cpu().numpy(), samplerate=16000)
            scores.append(score_list[-1])
            asr.append(suc)
            steps.append(step)
        print("asr: {}, score: {} avg steps: {}".format(np.mean(asr)*100, np.mean(scores), np.mean(steps)))
        np.savez(os.path.join(log_path, 'explog.npz'), asr=asr, scores=scores, steps=steps)
        return np.mean(asr), np.mean(scores), np.mean(steps)

    def BGMAAllTgt(self, iter, beta):
        print('Start Testing Model {} for style attack'.format(self.srs.modeltype))
        print('Initilize music is No.0')
        print('Total Speaker num %d' % self.spknum)
        music = self.musics[0]
        attack_writer = tensorboardX.SummaryWriter('./attack_log')
        
        # initiliaze the inverse matrix for mel-filter bank
        self.get_mel_inv()

        # load encoder and decoder
        generator_path = './ckpt/gen_00600000.pt'
        generator_config = get_config('./config/pia2gui_nfft2048_mel256.yaml')
        trainer = MUNIT_Trainer(generator_config)
        state_dict = torch.load(generator_path)
        trainer.gen_a.load_state_dict(state_dict['a'])
        trainer.gen_b.load_state_dict(state_dict['b'])
        trainer.cuda()
        trainer.eval()
        log_path = os.path.join('./results/BGMA/', self.srs.modeltype)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        asr = []
        for i, tgt_spk in enumerate(range(self.spknum)):
            print("Attack spk %d" % (i + 1))
            adv_music, suc, _, _, _  = self.BGMA(music, trainer.gen_a, trainer.gen_a, iter, attack_writer,
                                                              beta, tgt_spk, i)
            music_path = os.path.join(log_path, 'adv_music_{}_spk{}.wav'.format(str(i), str(tgt_spk)))
            sf.write(music_path, adv_music.detach().cpu().numpy(), samplerate=16000)
            asr.append(suc)
        print("asr: {}".format(np.mean(asr)*100))
        return


parser = argparse.ArgumentParser(description='Speaker Recognition white-box attack')
parser.add_argument('--epsilon', default=0.008, type=float)
parser.add_argument('--beta', default=1, type=float)
parser.add_argument('--dataset', default='TIMIT', type=str)
parser.add_argument('--steps', default=600, type=int)
parser.add_argument('--exp', default="BGMA-rand", type=str)
parser.add_argument('--obj', default="ut", type=str)
parser.add_argument('--seed', help='random seed', type=int, default=1234)
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
config = Config()
config.set_dataset(args.dataset)


def main(config):
    config.initialize = 'random'
    epsilon = args.epsilon
    steps = args.steps
    for model in ['TDNN', 'Dense', 'ECAPA']:
        print("Base Model:", model)
        config.modeltype = model
        srs = SRS(config)
        attacker = Attacker(srs)
        if args.exp == 'mupgd':
            attacker.clipMusicSlice(mode='r')
            attacker.MusicAttackDirectRandtgt(epsilon, steps, args.beta, 'linf')
        elif args.exp == 'BGMA-rand':
            attacker.clipMusicSlice(mode='r')
            attacker.BGMARandTgt(600, args.beta)
        elif args.exp == 'BGMA-all':
            attacker.clipMusicSlice(mode='r')
            attacker.BGMAAllTgt(600, 1)


if __name__ == '__main__':
    main(config)
