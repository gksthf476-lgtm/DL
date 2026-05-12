# DL

# Credit Score Classification using Deep Learning

## 신용점수 분류 딥러닝 프로젝트

본 프로젝트는 Kaggle의 **Credit Score Classification Dataset**을 활용하여 고객의 금융 정보를 기반으로 신용점수를 분류하는 딥러닝 기반 다중분류 프로젝트이다. 고객의 소득, 부채, 신용카드 이용 정보, 연체 이력, 대출 유형 등 다양한 금융 데이터를 바탕으로 신용점수를 `Good`, `Poor`, `Standard` 3개 클래스로 분류하는 것을 목표로 하였다. 단순히 모델 성능을 높이는 것뿐만 아니라, 데이터 전처리, Feature Engineering, 모델 구조 개선, 성능 비교 과정을 통해 딥러닝 모델의 성능 개선 방법을 이해하는 데 중점을 두었다.

---

## 프로젝트 정보

| 항목 | 내용 |
|---|---|
| 프로젝트명 | Credit Score Classification using Deep Learning |
| 데이터 출처 | Kaggle Credit Score Classification Dataset |
| 데이터 크기 | 100,000 rows × 28 columns |
| 타겟 변수 | Credit_Score |
| 문제 유형 | 다중분류 |
| 분류 클래스 | Good, Poor, Standard |
| 최종 모델 | MLP 기반 딥러닝 모델 |
| 최종 성능 | Validation Accuracy 0.7664 / Macro F1-score 0.7565 |

---

## 사용 기술

| 구분 | 기술 |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow, Keras |
| Model | MLP, Batch Normalization, Dropout, EarlyStopping, ReduceLROnPlateau |

---

## 데이터 전처리

먼저 개인 식별에 가까운 `ID`, `Customer_ID`, `Name`, `SSN` 컬럼은 신용점수 예측에 직접적인 의미가 없다고 판단하여 제거하였다. 이후 수치형 컬럼에 포함될 수 있는 특수문자, 공백, `_` 등을 정리하고 `float` 형태로 변환하였다. 금융 데이터에서는 고소득, 고부채, 고이용률 고객처럼 극단값이 실제 의미 있는 신호일 수 있기 때문에 IQR 기준으로 이상치를 무조건 제거하지 않았다. 대신 음수 소득, 음수 부채, 음수 연체일수처럼 현실적으로 불가능한 값만 결측치로 처리하는 방향으로 접근하였다.

범주형 변수에서는 `_______`, `_`, `!@9#%8`, `nan`, `NaN`, `None`, 빈 문자열과 같은 비정상적인 값을 `Unknown`으로 통일하였다. 또한 `Type_of_Loan` 컬럼은 하나의 셀에 여러 대출 유형이 함께 들어 있는 다중 범주형 변수였기 때문에 단순 원-핫 인코딩 대신 주요 대출 유형별 포함 여부를 0/1 변수로 변환하였다. 이를 통해 `Loan_Auto_Loan`, `Loan_Credit_Builder_Loan`, `Loan_Debt_Consolidation_Loan`, `Loan_Home_Equity_Loan`, `Loan_Mortgage_Loan`, `Loan_Not_Specified`, `Loan_Payday_Loan`, `Loan_Personal_Loan`, `Loan_Student_Loan` 변수를 생성하였다.

남아 있는 범주형 변수는 `pd.get_dummies()`를 활용하여 원-핫 인코딩하였고, 타겟 변수인 `Credit_Score`는 Label Encoding을 적용하였다. 최종적으로 `Good`은 0, `Poor`는 1, `Standard`는 2로 변환하였다. 전처리 후 최종 데이터 크기는 `X_encoded: (100000, 50)`, `y_encoded: (100000,)`가 되었다.

---

## 데이터 분할 및 스케일링

데이터는 Train과 Validation으로 8:2 비율로 분할하였다. 클래스 비율이 학습 데이터와 검증 데이터에 동일하게 유지될 수 있도록 `stratify=y_encoded`를 적용하였다. 분할 결과 `X_train`은 80,000개, `X_val`은 20,000개로 구성되었다. 금융 데이터는 극단값이 존재할 수 있기 때문에 평균과 표준편차 기반의 `StandardScaler`보다 이상치의 영향을 덜 받는 `RobustScaler`를 사용하였다. 단, 원-핫 인코딩된 범주형 변수는 이미 0과 1로 표현되어 있으므로 수치형 변수에만 스케일링을 적용하였다.

---

## Feature Selection

과적합을 줄이고 주요 변수를 확인하기 위해 `RandomForestClassifier` 기반 변수 중요도를 계산하였다. 모델에는 `n_estimators=200`, `random_state=42`, `n_jobs=-1`, `class_weight='balanced'`를 적용하였다. 상위 50개 변수를 선택하였으며, 인코딩 후 전체 Feature 수가 50개였기 때문에 실제 Feature 제거 효과보다는 변수 중요도 확인에 초점을 두었다.

주요 중요 변수로는 `Outstanding_Debt`, `Interest_Rate`, `Credit_Mix_Good`, `Credit_History_Age`, `Delay_from_due_date`, `Changed_Credit_Limit`, `Num_Credit_Inquiries`, `Total_EMI_per_month`, `Monthly_Balance`, `Month`, `Annual_Income`, `Monthly_Inhand_Salary` 등이 나타났다. 특히 `Outstanding_Debt`, `Interest_Rate`, `Credit_History_Age`와 같은 변수들은 신용점수와 밀접한 관련이 있는 금융 변수로 해석할 수 있다.

---

## 모델링

본 프로젝트에서는 정형 데이터를 다루기 때문에 CNN이나 RNN이 아닌 MLP, 즉 Multi-Layer Perceptron 모델을 사용하였다. MLP는 여러 수치형 및 범주형 Feature 간의 비선형 관계를 학습할 수 있어 신용점수 분류 문제에 적합하다고 판단하였다.

Baseline 모델은 `Dense(128) → BatchNormalization → Dropout(0.3) → Dense(64) → BatchNormalization → Dropout(0.2) → Dense(32) → Dropout(0.1) → Dense(3, softmax)` 구조로 설계하였다. Optimizer는 Adam을 사용하였고, 학습률은 `0.001`로 설정하였다.

Baseline 모델의 Validation Accuracy가 목표 기준인 0.75에 도달하지 못했기 때문에 모델 구조 확장, Batch Size 변경, Class Weight 적용, Learning Rate 감소, Dropout 비율 조정 등 여러 실험을 진행하였다.

---

## 실험 결과

| Experiment | 주요 변경 사항 | Validation Accuracy | Macro F1-score | Weighted F1-score |
|---|---|---:|---:|---:|
| Baseline | 기본 MLP 구조 | 0.7244 | 0.7074 | 0.7253 |
| Exp1 | Bigger MLP | 0.7366 | 0.7223 | 0.7375 |
| Exp2 | Batch Size 변경 | 0.7260 | 0.7083 | 0.7265 |
| Exp3 | Class Weight 적용 | 0.6884 | 0.6848 | 0.6899 |
| Exp5 | Bigger MLP + 낮은 Learning Rate + Dropout 조정 | 0.7664 | 0.7565 | 0.7672 |

실험 결과, 단순히 Batch Size를 변경하거나 Class Weight를 적용하는 것만으로는 성능이 개선되지 않았다. 오히려 Class Weight를 적용한 Exp3에서는 Validation Accuracy가 0.6884로 하락하였다. 반면 Exp5에서는 확장된 MLP 구조를 유지하면서 Learning Rate를 낮추고 Dropout 비율을 조정한 결과 Validation Accuracy 0.7664, Macro F1-score 0.7565를 달성하였다.

---

## 최종 모델

최종 모델은 Exp5 모델로 선정하였다. Exp5 모델은 더 깊고 넓은 MLP 구조를 사용하되, 과적합을 방지하기 위해 Batch Normalization과 Dropout을 함께 적용하였다. 또한 Learning Rate를 `0.0005`로 낮추어 학습을 더 안정적으로 진행하였다.

최종 모델 구조는 다음과 같다.

```python
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

