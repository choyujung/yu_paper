### 연극 포스터 이미지를 활용한 흥행예측연구

#### 프로젝트 설명 

포스터 이미지 데이터를 활용하여 CNN 모델과 회귀분석 방법을 사용한 관객수 예측 연구 입니다. 

#### 수행 기간 
20.06 ~ 21.03

#### 수행 역할 
- 데이터 수집
- 데이터 전처리
- 모델 구현
- 연구 논문 작성 

#### 사용 툴
- python, Tensorflow, scikit-learn

#### 연구 상세 내용
아래 URL을 통해 연구 배경과 연구 방법에 대한 상세 내용을 확인하실 수 있습니다. 

https://malleable-slayer-2fc.notion.site/34b527647fdc4d12a9eb1f210b16b4c7?pvs=4

---
#### 파일별 설명 

poster_images : 영화 포스터 이미지 수집 코드

이미지 학습에 사용된 딥러닝 모델 : c, resnet50, vgg

0728_f1score_versioin.ipynb : 모델 성능을 f1-score로 평가하기 위한 코드

0831_accuracy_save.ipynb : 모델 성능을 accuarcy 로 저장하고 평가

OCR vision api_0831.ipynb : 포스터 이미지에 포함되어 있는 글자를 추출하기 위해 OCR api를 적용한 코드

mtcnn.ipynb : 포스터 이미지 내의 인물 수를 추출하기 위해 mtcnn 모델을 활용하여 포스터별 인물 수 추출 코드

inception_v3.py :  inception-v3 모델 학습 코드 

resnet50.py : resnet50 모델 학습 코드 

vgg_model.py : vgg 모델 학습 코드
