import os
import numpy as np
import torch

from pytorch.nn.models import Tdnn, EcapaTdnn, DenseTdnn
from utils import getMel

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class SpeakerNet():

    def __init__(self, modeltype, config, num_classes=504):
        if modeltype == 'TDNN':
            self.net = Tdnn(feat_size=40, xvec_size=256, num_classes=num_classes).to(device)
        elif modeltype == 'ECAPA':
            self.net = EcapaTdnn(feat_size=40, xvec_size=192, num_classes=num_classes).to(device)
        elif modeltype == 'Dense':
            self.net = DenseTdnn(feat_size=40, xvec_size=256, num_classes=num_classes).to(device)
        self.net.load_state_dict(torch.load(os.path.join(config.preTrain_path, '{}.pth'.format(modeltype))))
        self.net.eval()

    def getEmbedding(self, utter):
        mel = getMel(utter)
        mel = mel.transpose(0, 1)
        mel = torch.tensor(mel).unsqueeze(0).float().to(device)
        # [batch, frames, n_mels]
        with torch.no_grad():
            ebd = self.net(mel)
        return ebd.squeeze().cpu().numpy()

    def getEmbeddingGC(self, utter):
        mel = getMel(utter)
        mel = mel.transpose(0, 1).unsqueeze(0).float()
        # mel = torch.tensor(mel).unsqueeze(0).float().to(device)
        # [batch, frames, n_mels]
        ebd = self.net(mel)
        return ebd.squeeze()

    def getEmbeddingBatch(self, utterbatch):
        outputs = []
        for u in utterbatch:
            u = torch.from_numpy(u).cuda()
            m = getMel(u)
            m = m.transpose(1, 0)
            with torch.no_grad():
                ebds = self.net(m.unsqueeze(0))
            outputs.append(ebds.squeeze().detach().cpu())
        outputs = torch.stack(outputs, dim=0)
        return outputs.numpy()

    def getBatchCenter(self, utterbatch):
        ebdbatch = self.getEmbeddingBatch(utterbatch)
        ebd = ebdbatch.mean(axis=0)
        ebd = ebd / np.linalg.norm(ebd)
        return ebd


