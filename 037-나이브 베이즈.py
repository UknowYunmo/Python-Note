import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas  as pd

# 1. 데이터 준비
col_names = ['sepal-length', 'sepal-width','petal-length', 'petal-width','Class'] 

# csv 파일에서 DataFrame을 생성
dataset = pd.read_csv('c:\\data\\iris2.csv', encoding='UTF-8', header=None, names=col_names)
#print(dataset) 


# DataFrame 확인
print(dataset.shape) # (row개수, column개수)
print(dataset.info()) # 데이터 타입, row 개수, column 개수, 컬럼 데이터 타입
print(dataset.describe()) # 요약 통계 정보 
print(dataset.iloc[0:5]) # dataset.head()
print(dataset.iloc[-5:]) # dataset.tail()

 
# X = 전체 행, 마지막 열 제외한 모든 열 데이터 -> n차원 공간의 포인트
X = dataset.iloc[:,:-1].to_numpy() # DataFrame을 np.ndarray로 변환
#print(X)

 
# 전체 데이터 세트를 학습 세트(training set)와 검증 세트(test set)로 나눔
# y = 전체 행, 마지막 열 데이터
y = dataset.iloc[:, 4].to_numpy()
#print(y)

 
# 데이터 분리
from sklearn.model_selection import train_test_split


# 전체 데이터 세트를 학습 세트(training set)와 검증 세트(test set)를 8:2로 나눔
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 10)
print(len(X_train), len(X_test))

print(X_train[:3])
print(y_train[:3])
 

#데이터 정규화 수행
# 1. 스케일(scale) : 평균0, 표준편차1인 데이터로 정규화
# 2. min/max : 0~1 사이의 숫자로 변경                   
from sklearn import preprocessing
X=preprocessing.StandardScaler().fit(X).transform(X)
from sklearn.model_selection import train_test_split              
# 훈련 데이터 70, 테스트 데이터 30으로 나눈다.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.3, random_state = 10)
# random_state=10은 R에서 seed값 설정과 동일
print(X_train.shape) # (398, 30)
print(y_train.shape) # (398, )

# 스케일링(z-score 표준화 수행 결과 확인)
for col in range(4):
    print(f'평균 = , 표준편차= ')

for col in range(4):
    print(f'평균 = , 표준편차= ')

# 4. 학습/예측(Training/Pradiction)
from sklearn.naive_bayes import BernoulliNB # 베르누이 나이브베이즈
from sklearn.naive_bayes import GaussianNB # 가우시안 나이브베이즈
#model = GaussianNB() # 가우시안 모델 선택 - 연속형 자료
#model = GaussianNB(var_smoothing=1e-09) # 라플라스 값 : 10의 -9승 

#model=BernoulliNB() # 베르누이 모델 선택 - 이진형 자료
model=BernoulliNB(alpha=0.1)
model.fit( X_train, y_train ) 

# 예측
y_pred= model.predict(X_test)
print(y_pred)
 
#5. 모델 평가
from sklearn.metrics import confusion_matrix
conf_matrix= confusion_matrix(y_test, y_pred)
print(conf_matrix) 

# 대각선에 있는 숫자가 정답을 맞춘 것, 그 외가 틀린 것
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report) 

# 이원 교차표 보는 코드
from sklearn import metrics
naive_matrix = metrics.confusion_matrix(y_test,y_pred)
print(naive_matrix) 

# 정확도 확인하는 코드
from sklearn.metrics import accuracy_score

accuracy = accuracy_score( y_test, y_pred)
print(accuracy)

#model=BernoulliNB() (베르누이 모델을) 사용했을 때 accuracy는 0.7333
#model=GaussianNB() (가우시안 모델을) 사용했을 때 accuracy는 1.0이 나왔다.