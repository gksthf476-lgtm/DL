# 딥러닝 컴페티션

# Credit Score Classification using Deep Learning

## 1. 프로젝트 개요

본 프로젝트는 Kaggle의 **Credit Score Classification Dataset**을 활용하여 고객의 신용점수를 분류하는 딥러닝 기반 다중분류 프로젝트이다.

고객의 소득, 부채, 신용카드 이용 정보, 연체 이력, 대출 유형 등 다양한 금융 데이터를 바탕으로 신용점수를 다음 3개 클래스로 분류하는 것을 목표로 한다.

- Good
- Poor
- Standard

본 프로젝트의 주요 목표는 단순히 모델 성능을 높이는 것뿐만 아니라, 데이터 전처리, feature engineering, 모델 구조 개선, 성능 비교 과정을 통해 딥러닝 모델의 성능 개선 방법을 이해하는 것이다.

---

## 2. 데이터셋

- 데이터 출처: Kaggle Credit Score Classification Dataset
- 데이터 크기: 100,000 rows × 28 columns
- 타겟 변수: `Credit_Score`

### 타겟 클래스 분포

| Class | 비율 |
|---|---:|
| Standard | 53.17% |
| Poor | 29.00% |
| Good | 17.83% |

클래스 분포를 확인한 결과 `Standard` 클래스의 비중이 가장 높고, `Good` 클래스의 비중이 가장 낮았다. 따라서 모델 평가 시 Accuracy뿐만 아니라 Macro F1-score도 함께 확인하였다.

---

## 3. 사용 기술

### Language & Library

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras

### Modeling

- MLP, Multi-Layer Perceptron
- Batch Normalization
- Dropout
- EarlyStopping
- ReduceLROnPlateau

---

## 4. 데이터 전처리

### 4.1 불필요한 컬럼 제거

개인을 식별하는 데 사용되는 컬럼은 모델 학습에 불필요하다고 판단하여 제거하였다.

제거한 컬럼은 다음과 같다.


```python
ID
Customer_ID
Name
SSN
```


### 4.2 수치형 컬럼 정리  

일부 수치형 컬럼에는 _, 공백, 특수문자 등이 섞여 있을 가능성이 있으므로 문자열을 정리한 뒤 float 형태로 변환하였다.
금융 데이터에서는 고소득, 고부채, 고이용률 고객이 실제로 존재할 수 있으므로 IQR 기준 이상치를 일괄 제거하지 않았다. 대신 음수 소득, 음수 부채, 음수 연체일수처럼 현실적으로 불가능한 값만 결측치로 처리하는 방식을 고려하였다.

-> 최종적으로 수치형 컬럼에서 문자 변환 실패나 현실적으로 불가능한 값은 발견되지 않아 추가 결측치 처리는 진행하지 않았다.  

### 4.3 범주형 변수 처리
범주형 변수에서 비정상적인 문자열은 Unknown으로 통일하였다.

처리한 이상 범주값 예시는 다음과 같다.

_
!@9#%8
nan
NaN
None
''

### 4.4 Type_of_Loan 변수 처리

Type_of_Loan 컬럼은 하나의 셀에 여러 대출 유형이 함께 들어 있는 다중 범주형 변수이다.

따라서 단순 원-핫 인코딩을 적용하면 조합별 컬럼이 과도하게 많아질 수 있으므로, 주요 대출 유형별 포함 여부를 0/1 변수로 변환하였다.


### 4.5 범주형 변수 인코딩
남아 있는 범주형 변수는 pd.get_dummies()를 활용하여 원-핫 인코딩하였다.

타겟 변수인 Credit_Score는 일반 feature가 아니므로 Label Encoding을 적용하였다.

- Good -> 0
- Poor -> 1
- Standard -> 2


---

## 5. 데이터 분할 및 스케일링
데이터는 Train과 Validation으로 8:2 비율로 분할하였다.

타겟 클래스 비율을 유지하기 위해 stratify=y_encoded를 적용하였다.

금융 데이터는 극단값이 존재할 수 있으므로 평균과 표준편차 기반의 StandardScaler보다 이상치의 영향을 덜 받는 RobustScaler를 사용하였다.

단, 원-핫 인코딩된 범주형 변수는 이미 0과 1로 표현되어 있으므로 수치형 변수에만 스케일링을 적용하였다.


---

## 6. Feature Selection
과적합을 줄이고 주요 변수를 확인하기 위해 RandomForestClassifier 기반 변수 중요도를 계산하였다.

상위 50개 변수를 선택하였으며, 인코딩 후 전체 feature 수가 50개였기 때문에 실제 feature 제거 효과보다는 변수 중요도 확인에 초점을 두었다.

주요 중요 변수는 다음과 같다.

<img width="923" height="528" alt="스크린샷 2026-05-12 154331" src="https://github.com/user-attachments/assets/f46231ab-7570-4f74-835d-53bec751f2ea" />


-> Month 변수도 상위 중요도에 포함되었지만, 이는 신용 행동 자체라기보다는 관측 시점에 따른 정보 변화가 반영된 결과일 수 있으므로 보조적인 시간 정보로 해석하였다.


---

## 7. 모델링

본 프로젝트에서는 정형 데이터를 다루기 때문에 CNN이나 RNN이 아닌 MLP 모델을 사용하였다.

MLP는 여러 feature 간의 비선형 관계를 학습할 수 있어 신용점수 분류 문제에 적합하다고 판단하였다.

---

## 8. 실험 과정

### 8.1 Baseline Model

Baseline 모델은 다음과 같은 구조로 설계하였다.

Dense(128) -> BatchNormalization -> Dropout(0.3)
Dense(64)  -> BatchNormalization -> Dropout(0.2)
Dense(32)  -> Dropout(0.1)
Dense(3)   -> Softmax

Optimizer는 Adam을 사용하였고, 학습률은 0.001로 설정하였다.


### 8.2 성능 개선 실험

<img width="350" height="471" alt="스크린샷 2026-05-12 154024" src="https://github.com/user-attachments/assets/158296aa-b547-4d0c-b081-5d22492f3578" />


Baseline 모델의 Validation Accuracy가 목표 기준인 0.75보다 낮게 나왔기 때문에 다음과 같은 개선 실험을 진행하였다.

(1) 모델 구조 확장
(2) Batch size 변경
(3) Class weight 적용
(4) Learning rate 감소 및 Dropout 비율 조정

---

## 9. 실험 결과

| Experiment | 주요 변경 사항                                   | Validation Accuracy | Macro F1-score | Weighted F1-score |
| ---------- | ------------------------------------------ | ------------------: | -------------: | ----------------: |
| Baseline   | 기본 MLP 구조                                  |              0.7244 |         0.7074 |            0.7253 |
| Exp1       | Bigger MLP                                 |              0.7366 |         0.7223 |            0.7375 |
| Exp2       | Batch size 변경                              |              0.7260 |         0.7083 |            0.7265 |
| Exp3       | Class weight 적용                            |              0.6884 |         0.6848 |            0.6899 |
| Exp5       | Bigger MLP + 낮은 Learning Rate + Dropout 조정 |              0.7664 |         0.7565 |            0.7672 |


## 10. 최종 모델

최종 모델은 Exp5 모델로 선정하였다.

Exp5 모델은 Exp1의 확장된 MLP 구조를 유지하면서 learning rate를 낮추고 Dropout 비율을 조정한 모델이다.

<최종 모델 구조>
Dense(256, activation='relu')
BatchNormalization()
Dropout(0.2)

Dense(128, activation='relu')
BatchNormalization()
Dropout(0.15)

Dense(64, activation='relu')
BatchNormalization()
Dropout(0.1)

Dense(32, activation='relu')
Dropout(0.05)

Dense(3, activation='softmax')

<학습 설정>
optimizer = Adam(learning_rate=0.0005)
loss = sparse_categorical_crossentropy
metrics = accuracy
epochs = 80
batch_size = 64

+ 과적합 방지를 위해 EarlyStopping을 적용하였고, validation loss가 정체될 경우 learning rate를 줄이기 위해 ReduceLROnPlateau를 사용하였다.

---


## 11. 최종 모델 성능

| Metric                       |  Score |
| ---------------------------- | -----: |
| Validation Loss              | 0.5517 |
| Validation Accuracy          | 0.7664 |
| Validation Macro F1-score    | 0.7565 |
| Validation Weighted F1-score | 0.7672 |



| Class    | Precision | Recall | F1-score | Support |
| -------- | --------: | -----: | -------: | ------: |
| Good     |      0.68 |   0.77 |     0.72 |   3,566 |
| Poor     |      0.75 |   0.78 |     0.77 |   5,799 |
| Standard |      0.81 |   0.76 |     0.78 |  10,635 |




최종 모델은 Validation Accuracy 0.7664, Macro F1-score 0.7565를 기록하여 목표 기준인 0.75를 달성하였다.

특히 Macro F1-score도 0.75 이상으로 나타나, 단순히 다수 클래스인 Standard에만 치우친 모델이 아니라 Good, Poor, Standard 세 클래스 전반에서 비교적 균형 있는 분류 성능을 보인 것으로 해석할 수 있다.


---


## 12. 결론

본 프로젝트에서는 신용점수 분류 문제를 해결하기 위해 금융 고객 데이터를 전처리하고, MLP 기반 딥러닝 모델을 구축하였다.

Baseline 모델의 성능은 목표 기준에 도달하지 못했지만, 모델 구조 확장, learning rate 조정, Dropout 비율 조정 등의 실험을 통해 최종적으로 Validation Accuracy 0.7664를 달성하였다.

가장 효과적이었던 개선 방법은 단순히 batch size를 변경하거나 class weight를 적용하는 것이 아니라, MLP 구조를 확장하고 learning rate와 Dropout 비율을 조정하는 것이었다.

이를 통해 딥러닝 모델의 성능은 단일 요소 하나만으로 개선되기보다, 모델 구조와 학습 전략을 함께 조정해야 향상될 수 있음을 확인하였다.


---


## 13. 향후 개선 방향

향후에는 다음과 같은 방법을 추가로 적용해볼 수 있다.

- 파생변수 생성
- Hyperparameter tuning
- XGBoost, CatBoost 등 머신러닝 모델과 성능 비교
- SHAP 기반 모델 해석
- 클래스별 오분류 패턴 분석
- Test 데이터 기반 최종 제출 성능 검증
