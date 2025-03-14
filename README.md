# Background Music Attack
Demos and PyTorch implementation of Background Music Attack (BGMA) in our paper ["Fooling Speaker Identification Systems with Adversarial Background Music"](https://www.isca-archive.org/interspeech_2023/zuo23_interspeech.html) (INTERSPEECH 2023).

## Prerequisites
* Python (3.6.9)
* Pytorch (1.2.0)
* SoundFile (0.10.3)
* librosa (0.8.1)
* CUDA
* numpy
* pydub
* tqdm


## Demos of Adversarial Music
### The following are the adversarial music generated by the migration of PGD. There are two examples for each target model: X-vector, D-TDNN, and ECAPA. The targeted model is indicated by the file name.



https://github.com/tartarleft/BGMA/assets/22472110/d1363ad0-42ab-4684-8804-7487dd622b23


https://github.com/tartarleft/BGMA/assets/22472110/d8b23215-1067-44e5-84cf-8bad17dd3e7c


https://github.com/tartarleft/BGMA/assets/22472110/32d0065d-25ab-477e-a250-3d4fa8e09b7f


https://github.com/tartarleft/BGMA/assets/22472110/fc5ef24b-b1a1-4e0c-b5fa-da8a3d5c6b18


https://github.com/tartarleft/BGMA/assets/22472110/d8592ebf-9c6d-4f83-99c8-3048fb2fbcb4


https://github.com/tartarleft/BGMA/assets/22472110/2cc4f867-0caa-429a-b1b6-cd444c439575

### The following are the adversarial music generated by BGMA, which is more auditorily natural with less white-noise-like sound in the music.



https://github.com/tartarleft/BGMA/assets/22472110/c9a145e7-27e3-4b7e-85ba-6023bd15e5d8


https://github.com/tartarleft/BGMA/assets/22472110/3570b628-a6b2-445c-95a1-939da6ffce27


https://github.com/tartarleft/BGMA/assets/22472110/c0853631-5a1b-4294-b121-e79d1c19fd83


https://github.com/tartarleft/BGMA/assets/22472110/658620ac-d438-46a5-9d15-771b9463b9c5


https://github.com/tartarleft/BGMA/assets/22472110/35340a68-a1be-4460-aa17-640de985d60f


https://github.com/tartarleft/BGMA/assets/22472110/9f403029-962b-4856-8f64-05407cbdcb8d




## Pretrained Models

We provide the pretrained SI models and pretrained music model in the [link](https://drive.google.com/file/d/1HImd_K88Q7cLkRowbDV-CpItgsjnlF3B/view?usp=drive_link). To run this project, unzip the 'ckpt' folder into the root directory of the project.
## Data
We provide the test waves on TIMIT and the enrolled speaker embeddings in [link](https://drive.google.com/file/d/1lnlc32JP2fpRWDkrO7pt9kgXOLbcqu2J/view?usp=sharing). To run this project,, unzip the 'data' folder into the root directory of the project. To reimplement the experiment in our paper, you also have to download the [MUSAN](https://www.openslr.org/17/) dataset and use the classical-HD subset as the initialization music. You can download the dataset into './data' or modify the corresponding path in './config/basic.json'.
 
## Usage
To run the PGD attack on music wave, run:
```
python Attacker-BGM.py --exp mupgd
```
To run BGMA on all the initialization muisc, run:
```
python Attacker-BGM.py --exp BGMA-rand
```

To run BGMA on one slices of muisc for all the target speakers, run:
```
python Attacker-BGM.py --exp BGMA-all
```

## Reference
[Play-As-You-Like](https://github.com/ChienYuLu/Play-As-You-Like-Timbre-Enhanced-Multi-modal-Music-Style-Transfer)

