## **Getting Started**
### **Dependency**
- This work was tested with PyTorch 1.10.1, CUDA 11.4, python 3.6 and Ubuntu 18.04.
  You may need pip3 install torch==1.10.1.
- requirements : lmdb, pillow, torchvision, nltk, natsort, jamo, fire, opencv-python

pip3 install lmdb pillow torchvision nltk natsort

pip3 install opencv-python

pip3 install jamo

pip3 install torch==1.10.1


### **Crop AIHUB dataset for training and Create LMDB**
<https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=105>

AIHUB데이터셋 압축해제, 저장 경로를 dataset_path로 넘겨주어 crop_dataset.py 실행

python3 crop_dataset.py

/data/
<pre>
ㄴ img
ㄴ label
ㄴ Validation
	ㄴ img
	    ㄴ 01.총류
	        ㄴ책표지_총류_002109.jpg
			⋮
	    ㄴ01.가로형간판
    ㄴ02.철학
		⋮
	ㄴ label
	    ㄴ1.간판
	    ㄴ2.책표지
	        ㄴ01.총류
	            ㄴ책표지_총류_002109.json
		⋮
	        ㄴ02.철학
		⋮
</pre>


### AIHUB의 Training dataset과 validation dataset 합치기

python3 merge.py 

### 자음 파인튜닝을 위해 합친 파일에서 ‘ㄲ’,’ㄸ’,’ㅃ’,’ㅆ’,’ㅉ’, ‘ㄳ’, ‘ㄵ’, ‘ㄶ’, ‘ㄺ’, ‘ㄻ’, ‘ㄼ’, ‘ㄽ’, ‘ㄾ’, ‘ㅀ’, ‘ㅄ’ 를 포함하는 이미지경로-라벨 쌍만 저장

python3 extract_text.py

### 대회에서 제공하는 train.csv로 저장

python3 create_gt_competition.py
<pre>
data
ㄴ train
    ㄴ cropped images
ㄴ validation
    ㄴ cropped images
ㄴ train_competition
    ㄴ competition train images
ㄴ gt_train.txt
ㄴ gt_valid.txt
ㄴ gt_merge.txt
ㄴ gt_jaeum.txt
ㄴ gt_competition.txt
</pre>

### LMDB로 변환

#### 1. AIHhub train + validation 합친 dataset을 lmdb로 변환

python3 create_lmdb_dataset.py --inputPath data/ --gtFile data/gt_merge.txt --outputPath data_lmdb_training

#### 2. 대회 train dataset을 lmdb로 변환

python3 create_lmdb_dataset.py --inputPath data/ --gtFile data/gt_competition.txt --outputPath data_lmdb_validation

#### 3. AIHub train + validation dataset에서 된소리,겹받침있는 image만 추출한 dataset을 lmdb로 변환

python3 create_lmdb_dataset.py --inputPath data/ --gtFile data/gt_jaeum.txt --outputPath data_lmdb_training_jaeum

## **Training and Submission**
### 1. Train TPS-SENet [1, 2, 5, 3]-BiLSTM-Attn model

python3 train.py --train_data data_lmdb_training --valid_data data_lmdb_validation --Transformation TPS --FeatureExtraction SENet --SequenceModeling BiLSTM --Prediction Attn --batch_size 52 --lr 1 --num_iter 67000 --manualSeed 1111

### 2. Train TPS-SENet_Large [2, 3, 7, 4]-BiLSTM-Attn model

python3 train.py --train_data data_lmdb_training --valid_data data_lmdb_validation --Transformation TPS --FeatureExtraction SENetL --SequenceModeling BiLSTM --Prediction Attn --batch_size 44 --lr 1 --num_iter 55000 --manualSeed 6

### 3. 2번에서 나온 model을 된소리, 겹받침 fine tuning

python3 train.py --train_data  data_lmdb_training_jaeum --valid_data data_lmdb_validation --saved_model SENetL.pth --Transformation TPS --FeatureExtraction SENetL --SequenceModeling BiLSTM --Prediction Attn --batch_size 44 --lr 0.3 --num_iter 1000 --manualSeed 6

### 4. Use  create_submission.py to create a submission file

if you want to use Pretrained files.
<https://drive.google.com/drive/folders/1JsWGSfR3_wUUS_3fHz1iBqCCL9J1DvjY?usp=sharing>


python3 create_submission.py --exp_name result --model1 SENetL_Jaeum.pth --model2 SENet.pth --model3 SENetL.pth --Transformation TPS --SequenceModeling BiLSTM --Prediction Attn
<br>you can change --image_folder (default='test') to set input test_data path
### **Arguments**
- --train_data: folder path to training lmdb dataset.
- --valid_data: folder path to validation lmdb dataset.
- --eval_data: folder path to evaluation (with test.py) lmdb dataset.
- --select_data: select training data.
- --data_filtering_off: skip [data filtering](https://github.com/clovaai/deep-text-recognition-benchmark/blob/f2c54ae2a4cc787a0f5859e9fdd0e399812c76a3/dataset.py#L126-L146) when creating LmdbDataset.
- --Transformation: select Transformation module [None | TPS].
- --FeatureExtraction: select FeatureExtraction module [VGG | RCNN | ResNet  | SENet  | SENetL].
- --SequenceModeling: select SequenceModeling module [None | BiLSTM].
- --Prediction: select Prediction module [CTC | Attn].
- --saved_model: assign saved model to evaluation.


