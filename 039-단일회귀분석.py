import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로드, 확인
df = pd.read_csv('c:\\data\\auto-mpg.csv', header=None)
df.columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin','name']
print(df.head())
pd.set_option('display.max_columns', 10) # 행 10개까지 출력
print(df.head())
 
# 2. 데이터 탐색
print(df.info())

horsepower에 전처리가 필요한 문자가 포함되어있어서 object 문자형으로 출력되었는데, 이를 수치형 데이터로 변경해야 한다. 

print(df['horsepower'].unique())

확인결과 '?' 라는 문자가 끼어있었다.

def isNumber(s):
  try:
    float(s)
    return True
  except ValueError:
    return False

for i in df['horsepower']:
    if isNumber(i)==False:
        print(i)

데이터가 대용량이라 눈으로 확인하기 힘든 경우는 위 방법을 사용하면 될 것 같다.

# 3. 결측치 처리
df['horsepower'].replace('?', np.nan, inplace=True) # '?'을 np.nan으로 변경
df.dropna(subset=['horsepower'], axis=0, inplace=True) # 누락 데이터 행(horsepower 기준)을 삭제
df['horsepower'] = df['horsepower'].astype('float') # 실수형 데이터만 남았기 때문에 object 타입이던 horsepower을 실수형으로 변환

# 4. 속성간 관계 살펴보기
ndf = df[['mpg', 'cylinders', 'horsepower', 'weight']]
ndf.plot(kind='scatter', x='weight', y='mpg', c='coral', s=10, figsize=(10, 5))
plt.show()
plt.close()


fig = plt.figure(figsize=(10, 5))   #  전체 그림판 가로 10, 세로 5로 잡아주고 
ax1 = fig.add_subplot(1, 2, 1)    #  첫번째 그림판 영역
ax2 = fig.add_subplot(1, 2, 2)    #  두번째 그림판 영역 
sns.regplot(x='weight', y='mpg', data=ndf, ax=ax1) # 회귀선 표시
sns.regplot(x='weight', y='mpg', data=ndf, ax=ax2, fit_reg=False) #회귀선 미표시
plt.show()
plt.close()


sns.jointplot(x='weight', y='mpg', data=ndf) # 회귀선 없음
sns.jointplot(x='weight', y='mpg', kind='reg', data=ndf) # 회귀선 표시
plt.show()
plt.close()


sns.pairplot(ndf) # 모든 경우의 수 그리기
plt.show()
plt.close()

# 5. 데이터셋 구분 - 훈련용(train data)/ 검증용(test data)

속성(변수) 선택
X=ndf[['weight']] #독립 변수 X
y=ndf['mpg'] #종속 변수 Y

# train data 와 test data로 구분(7:3 비율)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10) # seed : 10

print('train data 개수: ', len(X_train))
print('test data 개수: ', len(X_test))

# 6. 단순회귀분석 모형 - sklearn 사용

from sklearn.linear_model import LinearRegression
lr = LinearRegression() # 단순회귀분석 모형 객체 생성
lr.fit(X_train, y_train) # train data를 가지고 모형 학습

r_square = lr.score(X_test, y_test) # 학습을 마친 모형에 test data를 적용하여 결정계수(R-제곱) 계산
print(r_square) # 0.6822458558299325
print(lr.coef_) # 회귀식의 기울기
print(lr.intercept_) # 회귀식의 y절편

# 모형에 전체 X 데이터를 입력하여 예측한 값 y_hat을 실제 값 y와 비교
y_hat = lr.predict(X)
plt.figure(figsize=(10, 5)) # plot 사이즈 설정
ax1 = sns.distplot(y, hist=False, label="y")
ax2 = sns.distplot(y_hat, hist=False, label="y_hat", ax=ax1)
plt.show()
plt.close()

# 예측 성능을 높이기 위하여 단순 회귀 -> 다항 회귀

1. 단순회귀 :  독립변수 한개에 종속변수 한개 ( 선형 회귀선 )
2. 다항회귀 :  독립변수 한개에 종속변수 한개 ( 비선형 회귀선 )

# 다항식 변환 
poly = PolynomialFeatures(degree=2)               #2차항 적용
X_train_poly=poly.fit_transform(X_train)     #X_train 데이터를 2차항으로 변형

print(X_train.shape) #(274, 1)
print(X_train_poly.shape) #(274, 3)

# train data를 가지고 모형 학습
pr = LinearRegression()   
pr.fit(X_train_poly, y_train)

# 학습을 마친 모형에 test data를 적용하여 결정계수(R-제곱) 계산
X_test_poly = poly.fit_transform(X_test)       #X_test 데이터를 2차항으로 변형
r_square = pr.score(X_test_poly,y_test)
print(r_square) #0.7087009262975685

# train data의 산점도와 test data로 예측한 회귀선을 그래프로 출력 
y_hat_test = pr.predict(X_test_poly)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1)
ax.plot(X_train, y_train, 'o', label='Train Data')  # 데이터 분포
ax.plot(X_test, y_hat_test, 'r+', label='Predicted Value') # 모형이 학습한 회귀선
ax.legend(loc='best')
plt.xlabel('weight')
plt.ylabel('mpg')
plt.show()
plt.close()

# 모형에 전체 X 데이터를 입력하여 예측한 값 y_hat을 실제 값 y와 비교 
X_ploy = poly.fit_transform(X)
y_hat = pr.predict(X_ploy)

plt.figure(figsize=(10, 5))
ax1 = sns.distplot(y, hist=False, label="y")
ax2 = sns.distplot(y_hat, hist=False, label="y_hat", ax=ax1)
plt.show()
plt.close()