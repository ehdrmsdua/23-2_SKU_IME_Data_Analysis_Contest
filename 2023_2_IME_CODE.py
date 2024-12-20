#!/usr/bin/env python
# coding: utf-8

# In[1]:


## 1번 import
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')


# In[2]:


df = pd.read_csv('C:/Users/USER/OneDrive/바탕 화면/data/data.csv',encoding='cp949')


# In[3]:


pd.options.display.float_format ='{:.4f}'.format


# In[4]:


df.describe()


# In[5]:


##변환
df.dtypes
df['TAG_MIN'] =pd.to_datetime(df.TAG_MIN, format='%Y-%m-%d %H:%M:%S')


# In[6]:


df.dtypes


# In[7]:


df.isnull().sum()


# In[8]:


## 결측치 평균으로 처리
df = df.fillna(df.mean())


# In[9]:


df.isnull().sum()


# In[10]:


import matplotlib
matplotlib.rcParams['axes.unicode_minus'] =False
import matplotlib.pyplot as plt


# In[11]:


## 히트맵 작성
import seaborn as sns
plt.rcParams['font.size'] =20
plt.rcParams['font.family'] ='Malgun Gothic'
fig = plt.figure(figsize=(40,20))
cmap ='coolwarm'
target_cor = df.corr().loc[[ '건조 1존 OP', '건조 2존 OP', '건조로 온도 1 Zone',
 '건조로 온도 2 Zone', '세정기', '소입1존 OP', '소입2존 OP', '소입3존 OP', '소입4존 OP',
 '소입로 CP 값', '소입로 CP 모니터 값', '소입로 온도 1 Zone', '소입로 온도 2 Zone',
 '소입로 온도 3 Zone', '소입로 온도 4 Zone', '솔트 컨베이어 온도 1 Zone',
 '솔트 컨베이어 온도 2 Zone', '솔트조 온도 1 Zone', '솔트조 온도 2 Zone']]
mask = np.triu(np.ones_like(target_cor, dtype=np.bool))
sns.heatmap(target_cor, mask=mask,
 vmin=-1, vmax=1,
 cmap ='coolwarm', linewidths=.5,
 annot=True, fmt=".2f", xticklabels=False, yticklabels=False)
plt.xticks(np.arange(0.5, len(target_cor.columns)+0.5, 1), target_cor.columns, size =20, rotation=90)
plt.yticks(np.arange(0.5, len(target_cor.index)+0.5, 1), target_cor.index, size =20)
plt.tight_layout()


# In[12]:


## 상관관계 높은 변수 1개 드롭
df.drop(['소입로 CP 모니터 값'], axis=1, inplace=True)


# In[13]:


## 평균과 분산으로 나눔
df_stat = df.groupby(['배정번호']).agg(['mean', 'std'])
df_stat


# In[14]:


## 인덱스화 생성
chg_name = {'mean': '_Avg', 'std': '_Std'}
df_stat.columns = list(map(lambda x: x[0] + chg_name[x[1]], df_stat.columns))
df_stat.reset_index(drop=False, inplace=True)


# In[15]:


df_stat


# In[16]:


## 종속변수 불러오기
heat_result = pd.read_excel("C:/Users/USER/OneDrive/바탕 화면/교내 활동/2023-2학기 산업경영공학과 인공지능 및 데이터사이언스 경진대회/data/열처리_품질데이터.xlsx")
heat_result


# In[17]:


## 필요없는 column 날리기
heat_result.drop(['작업일', '공정명', '설비명', '양품수량'], axis=1, inplace=True)


# In[18]:


# 공정 데이터와 품질 데이터를 결합한다.
df_total = pd.merge(heat_result, df_stat, on='배정번호', how='left')


# In[19]:


df_total.info()


# In[20]:


## 불량률 변수 생성
df_total["불량률"] = round(df_total["불량수량"] / df_total["총수량"] *100, 3)


# In[21]:


plt.figure(figsize=(5,3))
plt.boxplot(df_total['불량률'], vert=False)
plt.title('불량률 boxplot')
plt.yticks([1], ["불량률"])
plt.tight_layout()


# In[22]:


df_total['불량률'].describe()


# In[23]:


df_total.loc[df_total['불량률'] >=0.046, '불량수준'] ='위험'
df_total.loc[(df_total['불량률'] <0.046) & (df_total['불량률'] >= 0.0127), '불량수준'] ='주의'
df_total.loc[df_total['불량률'] <0.0127, '불량수준'] ='안정'


# In[24]:


print(df_total.columns)


# In[25]:


#원-핫-인코딩
df_encoded = pd.get_dummies(df_total, columns=['불량수준'])


# In[26]:


X_num = df_total[['건조 1존 OP_Avg', '건조 1존 OP_Std', '건조 2존 OP_Avg',
'건조 2존 OP_Std','건조로 온도 1 Zone_Avg', '건조로 온도 1 Zone_Std',
'건조로 온도 2 Zone_Avg', '건조로 온도 2 Zone_Std', '세정기_Avg', '세정기_Std',
'소입1존 OP_Avg', '소입1존 OP_Std', '소입2존 OP_Avg', '소입2존 OP_Std',
'소입3존 OP_Avg', '소입3존 OP_Std', '소입4존 OP_Avg', '소입4존 OP_Std',
'소입로 CP 값_Avg', '소입로 CP 값_Std', '소입로 온도 1 Zone_Avg',
'소입로 온도 1 Zone_Std', '소입로 온도 2 Zone_Avg', '소입로 온도 2 Zone_Std',
'소입로 온도 3 Zone_Avg', '소입로 온도 3 Zone_Std', '소입로 온도 4 Zone_Avg',
'소입로 온도 4 Zone_Std', '솔트 컨베이어 온도 1 Zone_Avg',
'솔트 컨베이어 온도 1 Zone_Std', '솔트 컨베이어 온도 2 Zone_Avg',
'솔트 컨베이어 온도 2 Zone_Std', '솔트조 온도 1 Zone_Avg', '솔트조 온도 1 Zone_Std',
'솔트조 온도 2 Zone_Avg', '솔트조 온도 2 Zone_Std']]
y = df_encoded[['불량수준_안정','불량수준_위험','불량수준_주의']]


# In[27]:


df_total['불량수준'].value_counts()


# In[28]:


# 주석 처리 재실행 필요 df_encoded.drop(['배정번호', '불량수량', '총수량','불량률'], axis=1, inplace=True)
df_encoded.head()


# In[29]:


from sklearn.model_selection import train_test_split
X_train_select, X_test_select, y_train_select, y_test_select= train_test_split(X_num, y, test_size=0.3, random_state=1)


# In[30]:


from sklearn.preprocessing import MinMaxScaler
MMScale = MinMaxScaler()
X_train_scaled = MMScale.fit_transform(X_train_select)
X_test_scaled = MMScale.transform(X_test_select)


# In[31]:


##train 행 개수 확인하기
num_rows = X_train_scaled.shape[0]
print("개수 확인:", num_rows)


# In[32]:


#훈련데이터 독립변수 구성
mean_value = np.mean(X_train_scaled)
max_value = np.max(X_train_scaled)
min_value = np.min(X_train_scaled)

# 결과 출력
print("평균 값:", mean_value)
print("최대 값:", max_value)
print("최소 값:", min_value)


# In[33]:


print(X_train_scaled)


# In[34]:


#테스트 데이터 독립변수 구성
mean_value_2 = np.mean(X_test_scaled)
max_value_2 = np.max(X_test_scaled)
min_value_2 = np.min(X_test_scaled)

# 결과 출력
print("평균 값:", mean_value_2)
print("최대 값:", max_value_2)
print("최소 값:", min_value_2)


# In[35]:


# 훈련데이터 종속변수 구성
y_train_select.describe()


# In[36]:


# 테스트 데이터 종속변수 구성
y_test_select.describe()


# In[37]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


# In[38]:


pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled)


# In[56]:


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')


colors = y_train_select['불량수준_안정'] + 2 * y_train_select['불량수준_위험'] + 3 * y_train_select['불량수준_주의']
custom_colors = ['red', 'green', 'blue']  
markers = ['o', '^', 's']  

for class_label, (color, marker) in enumerate(zip(custom_colors, markers), start=1):
    class_indices = (colors == class_label)
    scatter = ax.scatter(
        X_train_pca[class_indices, 0],
        X_train_pca[class_indices, 1],
        X_train_pca[class_indices, 2],
        c=color,
        marker=marker,
        s=20,
        label=f'불량수준_{["안정", "위험", "주의"][class_label - 1]}'
    )
plt.rcParams.update({'font.size': 9})
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('PCA에 따른 종속변수 분포 (3D)')


ax.legend()
plt.show()


# In[40]:


# 3차원 pca 분산 보존율
explained_variance_ratio = pca.explained_variance_ratio_
print(explained_variance_ratio)


# In[41]:


# PCA 수행 (n_components를 2로 설정)
pca_2 = PCA(n_components=2)
X_train_pca_2 = pca_2.fit_transform(X_train_scaled)

# 불량수준에 따라 색상 지정
colors = y_train_select['불량수준_안정'] + 2 * y_train_select['불량수준_위험'] + 3 * y_train_select['불량수준_주의']
custom_colors = ['red', 'green', 'blue']  
markers = ['o', '^', 's']  

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

for class_label, (color, marker) in enumerate(zip(custom_colors, markers), start=1):
    class_indices = (colors == class_label)
    scatter = ax.scatter(
        X_train_pca_2[class_indices, 0],
        X_train_pca_2[class_indices, 1],
        c=color,
        marker=marker,
        s=20,
        label=f'불량수준_{["안정", "위험", "주의"][class_label - 1]}'
    )

ax.legend()
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_title('PCA에 따른 불량수준 분포 (2D)')

plt.show()


# In[42]:


# 2차원 주성분 분산 보존율
explained_variance_ratio2 = pca_2.explained_variance_ratio_
print(explained_variance_ratio2)


# In[43]:


# 3차원 주성분 히트맵
plt.matshow(pca.components_, cmap = 'viridis')
plt.yticks([0,1,2], ['PCA 1', 'PCA 2', 'PCA 3'])
plt.colorbar()
plt.xticks(range(len(X_train_select.columns)), X_train_select.columns, rotation = 60, ha ='left')
plt.xlabel('특성')
plt.ylabel('주성분')


# In[44]:


# 3차원 pca 가중치
print(pca.components_)


# In[45]:


# 2차원 pca 가중치 히트맵
plt.matshow(pca_2.components_, cmap = 'viridis')
plt.yticks([0,1], ['PCA 1', 'PCA 2'])
plt.colorbar()
plt.xticks(range(len(X_train_select.columns)), X_train_select.columns, rotation = 60, ha ='left')
plt.xlabel('특성')
plt.ylabel('주성분')


# In[46]:


# 2차원 pca 가중치
print(pca_2.components_)


# In[74]:


from sklearn.neighbors import KNeighborsClassifier
training_accu =[]
test_accu = []
find_neighbors = range(1,25)

for n_neighbors in find_neighbors :
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train_scaled, y_train_select)
    y_pred1 = knn_model.predict(X_train_scaled)
    y_pred2 = knn_model.predict(X_test_scaled)
    training_accu.append(np.mean(y_pred1.argmax(axis=1) == y_train_select.values.argmax(axis=1)))
    test_accu.append(np.mean(y_pred2.argmax(axis=1) == y_test_select.values.argmax(axis=1)))
plt.plot(find_neighbors, training_accu , label = "훈련 정확도")
plt.plot(find_neighbors, test_accu , label = "테스트 정확도")
plt.ylabel("정확도")
plt.xlabel("이웃 수")
plt.legend()


# In[79]:


#n 명으로 knn 돌렸을 때의 정확도
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train_scaled, y_train_select)

y_pred_test = knn_model.predict(X_test_scaled)
y_pred_train = knn_model.predict(X_train_scaled)

accuracy_train = np.mean(y_pred_train.argmax(axis=1) == y_train_select.values.argmax(axis=1))
print("훈련 정확도: {:.2f}".format(accuracy_train))

accuracy = np.mean(y_pred_test.argmax(axis=1) == y_test_select.values.argmax(axis=1))
print("테스트 정확도: {:.2f}".format(accuracy))


# In[80]:


print(y_pred_train)


# In[69]:


from sklearn.metrics import classification_report

# 다양한 이웃 수 시도
for n_neighbors in range(1, 26):
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train_scaled, y_train_select)
    
    y_pred_train = knn_model.predict(X_train_scaled)
    y_pred_test = knn_model.predict(X_test_scaled)
    
    accuracy_train = np.mean(y_pred_train.argmax(axis=1) == y_train_select.values.argmax(axis=1))
    accuracy_test = np.mean(y_pred_test.argmax(axis=1) == y_test_select.values.argmax(axis=1))
    
    # 기타 성능 지표 출력
    print(f"이웃 수: {n_neighbors}")
    print("훈련 데이터 성능:")
    print(classification_report(y_train_select.values.argmax(axis=1), y_pred_train.argmax(axis=1)))
    
    print("테스트 데이터 성능:")
    print(classification_report(y_test_select.values.argmax(axis=1), y_pred_test.argmax(axis=1)))
    print("-" * 50)


# In[50]:


y_train_select_1d = np.argmax(y_train_select.values, axis=1)
print(y_train_select_1d)
y_test_select_1d = np.argmax(y_test_select.values, axis=1)


# In[51]:


from sklearn.svm import SVC
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train_scaled, y_train_select_1d)


# In[52]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
C_values =  np.arange(0.1, 10, 0.5)
gamma_values = np.linspace(0.01, 1, 10)

# 변수 초기화
best_score = 0
best_params = {}


for C in C_values:
    for gamma in gamma_values:
        svm_model = SVC(C=C, gamma=gamma, kernel='rbf')
        svm_model.fit(X_train_scaled, y_train_select_1d)

        scores = cross_val_score(svm_model, X_train_scaled, y_train_select_1d, cv=5)
        mean_score = np.mean(scores)
        
        # 현재 매개변수로 얻은 점수가 더 높으면 갱신
        if mean_score > best_score:
            best_score = mean_score
            best_params = {'C': C, 'gamma': gamma}

# 최적의 매개변수 및 성능 출력
print("Best parameters: ", best_params)


# In[53]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 3. 모델 생성 및 학습
svm_model = SVC(kernel='rbf', C = 5.1 , gamma = 0.12)
svm_model.fit(X_train_scaled, y_train_select_1d)

# 5. 모델 평가
y_train_pred = svm_model.predict(X_train_scaled)
y_test_pred = svm_model.predict(X_test_scaled)

accuracy_train = accuracy_score(y_train_select_1d, y_train_pred)
accuracy_test = accuracy_score(y_test_select_1d, y_test_pred)

# 훈련 데이터에 대한 classification report 출력
print("Train Data Metrics:")
print(classification_report(y_train_select_1d, y_train_pred))

# 테스트 데이터에 대한 classification report 출력
print("\nTest Data Metrics:")
print(classification_report(y_test_select_1d, y_test_pred))


# In[54]:


#랜덤 바꿔해보기
X_train_select_r, X_test_select_r, y_train_select_r, y_test_select_r= train_test_split(X_num, y, test_size=0.2, random_state=20)
MMScale_r = MinMaxScaler()
X_train_scaled_r = MMScale_r.fit_transform(X_train_select_r)
X_test_scaled_r = MMScale_r.transform(X_test_select_r)
y_train_select_1d_r = np.argmax(y_train_select_r.values, axis=1)
y_test_select_1d_r = np.argmax(y_test_select_r.values, axis=1)

C_values =  np.arange(0.1, 10, 0.5)
gamma_values = np.linspace(0.01, 1, 10)

# 변수 초기화
best_score_r = 0
best_params_r = {}


for C in C_values:
    for gamma in gamma_values:
        svm_model_r = SVC(C=C, gamma=gamma, kernel='rbf')
        svm_model_r.fit(X_train_scaled_r, y_train_select_1d_r)

        scores = cross_val_score(svm_model_r, X_train_scaled_r, y_train_select_1d_r, cv=5)
        mean_score_r = np.mean(scores)
        
        # 현재 매개변수로 얻은 점수가 더 높으면 갱신
        if mean_score > best_score_r:
            best_score_R = mean_score_r
            best_params_r = {'C': C, 'gamma': gamma}

# 최적의 매개변수 및 성능 출력
print("Best parameters: ", best_params_r)



# In[55]:


#랜덤 바꿔 출력하기

svm_model_r = SVC(kernel='rbf', C = 9.6 , gamma = 1.0)
svm_model_r.fit(X_train_scaled_r, y_train_select_1d)

# 5. 모델 평가
y_train_pred_r = svm_model_r.predict(X_train_scaled_r)
y_test_pred_r = svm_model_r.predict(X_test_scaled_r)

# 훈련 데이터에 대한 classification report 출력
print("Train Data Metrics:")
print(classification_report(y_train_select_1d_r, y_train_pred_r))

# 테스트 데이터에 대한 classification report 출력
print("\nTest Data Metrics:")
print(classification_report(y_test_select_1d_r, y_test_pred_r))


# In[ ]:


gamma_valuess = np.linspace(0.01, 1, 100)
print(gamma_valuess)

