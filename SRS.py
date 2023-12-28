import os
import random
import pickle
import numpy as np
import torch
from SpeakerNet import SpeakerNet
from utils import CosineSimilarity, CosineSimilarityGC


class SRS():

    def __init__(self, config, modeltype=None, ebd_mean_num=5):
        if modeltype is not None:
            self.modeltype = modeltype
        else:
            self.modeltype = config.modeltype
        self.ebd_mean_num = ebd_mean_num
        if self.modeltype not in ['TDNN', 'Dense', 'ECAPA']:
            print('ModelType Not Defined')
            return 0
        if config.dataset == 'TIMIT':
            self.net = SpeakerNet(self.modeltype, config, num_classes=504)
        elif config.dataset == 'libri':
            self.net = SpeakerNet(self.modeltype, config, num_classes=522)
        self.spkebds = self.__initial_embeddings(config)
        self.spknum = len(self.spkebds['name'])


    def __initial_embeddings(self, config):
        os.makedirs(config.ebds_path, exist_ok=True)
        ebds_path = os.path.join(config.ebds_path, '{}_ebds.pkl'.format(self.modeltype))
        if os.path.exists(ebds_path):
            with open(ebds_path, 'rb') as f:
                ebds = pickle.load(f)
                print("ebds_path: ", ebds_path)
                print('Loading Embeddings for {} Speakers'.format(len(ebds['name'])))
                return ebds
        spkwavs = os.listdir(config.data_path)
        print('Generating Embeddings For {} Speakers'.format(len(spkwavs)))
        ebds = {}
        ebds['name'] = []
        ebds['ebds'] = []
        for spk in spkwavs:
            with open(os.path.join(config.data_path, spk), 'rb') as f:
                utterlist = pickle.load(f)
            ulist = random.sample(list(utterlist), self.ebd_mean_num)
            ebd = self.net.getBatchCenter(ulist)
            ebds['name'].append(spk)
            ebds['ebds'].append(ebd)
        with open(ebds_path, 'wb') as f:
            pickle.dump(ebds, f)
        return ebds

    def Avaliable(self, restQ=0):
        return self.__qtime + restQ < self.__maxqtime

    def CurQTimes(self):
        return self.__qtime

    def MaxQTimes(self):
        return self.__maxqtime

    def Add(self, a=1):
        # if a!= 1:
        # print('+{}'.format(a))
        self.__qtime += a

    def getSpkList(self):
        return self.spkebds['name']

    def Query(self, utter):
        cur_ebd = self.net.getEmbedding(utter)
        simmatrix = CosineSimilarity(cur_ebd, np.array(self.spkebds['ebds']))
        return np.argmax(simmatrix)

    def Querynp(self, utter):
        utter = torch.from_numpy(utter).cuda()
        cur_ebd = self.net.getEmbedding(utter)
        simmatrix = CosineSimilarity(cur_ebd, np.array(self.spkebds['ebds']))
        return np.argmax(simmatrix)

    def QuerySimGC(self, utter):
        cur_ebd = self.net.getEmbeddingGC(utter)
        ebds = torch.tensor(self.spkebds['ebds']).cuda()
        simmatrix = CosineSimilarityGC(cur_ebd, ebds)
        return simmatrix

    def QueryTgtUtterSimGC(self, utter, tgt_utters):
        cur_ebd = self.net.getEmbeddingGC(utter)
        tgt_ebds = [self.net.getEmbeddingGC(u) for u in tgt_utters]
        tgt_ebds = torch.stack(tgt_ebds)
        simmatrix = CosineSimilarityGC(cur_ebd, tgt_ebds)
        return simmatrix

    def QueryBatch(self, utters):
        ebds = self.net.getEmbeddingBatch(utters)
        ids = []
        for e in ebds:
            simmatrix = CosineSimilarity(e, np.array(self.spkebds['ebds']))
            ids.append(np.argmax(simmatrix))
        return ids

    def Reset(self, Qt=5000):
        self.__qtime = 0
        self.__maxqtime = Qt
