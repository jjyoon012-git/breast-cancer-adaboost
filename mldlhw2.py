import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def standardize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std

def to_pm1_labels(y):
    # sklearn breast_cancer: 0=benign, 1=malignant
    # {-1, +1}로 변환
    return np.where(y == 1, +1, -1)

# STEP 2. Decision Stump
class DecisionStump:
    def __init__(self):
        self.j = None        # feature index
        self.thr = None      # threshold
        self.polarity = 1    # +1 또는 -1

    def predict(self, X):
        xj = X[:, self.j]
        pred = np.ones(X.shape[0])
        # polarity * x < polarity * thr 면 -1, 아니면 +1
        pred[self.polarity * xj < self.polarity * self.thr] = -1
        return pred

# 직접 구현한 AdaBoost 
class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []

    def fit(self, X, y):
        n, d = X.shape

        # STEP 1. 가중치 초기화
        w = np.ones(n) / n
        self.stumps = []
        self.alphas = []

        for t in range(self.n_estimators):
            # STEP 2. Decision Stump
            stump, err = self._best_stump(X, y, w)

            # STEP 3. 분류기 가중치 계산 (오분류율 계산 포함)
            eps = 1e-12
            err = np.clip(err, eps, 1 - eps)
            alpha = 0.5 * np.log((1 - err) / err)

            # STEP 4. 샘플 가중치 업데이트
            pred = stump.predict(X)
            w = w * np.exp(-alpha * y * pred)
            w = w / w.sum()  # 정규화

            # 누적 저장
            self.stumps.append(stump)
            self.alphas.append(alpha)

    def _best_stump(self, X, y, w):
        n, d = X.shape
        best_err = 1e9
        best = DecisionStump()

        # 모든 feature에 대해 임계값 후보 탐색. 중간값을 사용했습니다.
        for j in range(d):
            xj = X[:, j]
            order = np.argsort(xj)
            xj_s = xj[order]
            y_s = y[order]
            w_s = w[order]

            uniq = np.unique(xj_s)
            if len(uniq) == 1:
                continue
            thresholds = (uniq[:-1] + uniq[1:]) / 2.0

            for pol in (+1, -1):
                for thr in thresholds:
                    pred = np.ones_like(y_s)
                    pred[pol * xj_s < pol * thr] = -1
                    err = np.sum(w_s[pred != y_s])
                    if err < best_err:
                        best_err = err
                        best.j = j
                        best.thr = thr
                        best.polarity = pol
        return best, best_err

    def predict_scores(self, X):
        # STEP 5. 최종 예측 결합: F(x) = sum_t alpha_t * h_t(x)
        score = np.zeros(X.shape[0])
        for stump, a in zip(self.stumps, self.alphas):
            score += a * stump.predict(X)
        return score

    def predict(self, X):
        # STEP 5. 최종 예측 결합 결과에 sign 적용
        return np.sign(self.predict_scores(X))


# Streamlit UI

st.set_page_config(page_title="AdaBoost (from scratch) - Breast Cancer", layout="wide")
st.title("AdaBoost (from scratch) with Decision Stumps — Breast Cancer")

with st.sidebar:
    st.header("Settings")
    n_estimators = st.slider("Number of weak learners (T)", 10, 100, 30, 5)
    test_size = st.slider("Test size ratio", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    show_data = st.checkbox("Show raw data (first 5 rows)", value=False)

# 데이터 로드
data = load_breast_cancer()
X_raw = data.data.astype(float)
y_raw = data.target.astype(int)  # 0=benign, 1=malignant
feature_names = list(data.feature_names)

df = pd.DataFrame(X_raw, columns=feature_names)
df["target(0=B,1=M)"] = y_raw
if show_data:
    st.subheader("Raw data (head)")
    st.dataframe(df.head())

# 라벨 {-1, +1}
y_pm1 = to_pm1_labels(y_raw)

# train/test split
X_tr, X_te, y_tr, y_te = train_test_split(
    X_raw, y_pm1, test_size=test_size, random_state=random_state, stratify=y_pm1
)

# 표준화
X_tr, X_te = standardize(X_tr, X_te)

# STEP 6. 평가
# T=1..n_estimators로 늘리며 Test Accuracy 곡선 기록
acc_list = []
pred_list = []
for T in range(1, n_estimators + 1):
    model = AdaBoost(n_estimators=T)
    model.fit(X_tr, y_tr)
    y_hat = model.predict(X_te)
    acc = (y_hat == y_te).mean()
    acc_list.append(acc)
    pred_list.append(y_hat)

# 결과 레이아웃
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Final Test Accuracy")
    st.write(f"T = **{n_estimators}** 일 때, Test Accuracy = **{acc_list[-1]*100:.2f}%**")

    # Confusion Matrix도 구현해봤습니다.
    y_hat_final = pred_list[-1]
    tn = np.sum((y_te == -1) & (y_hat_final == -1))
    tp = np.sum((y_te == +1) & (y_hat_final == +1))
    fn = np.sum((y_te == +1) & (y_hat_final == -1))
    fp = np.sum((y_te == -1) & (y_hat_final == +1))
    cm = pd.DataFrame([[tn, fp],[fn, tp]],
                      index=["True -1(B)", "True +1(M)"],
                      columns=["Pred -1(B)", "Pred +1(M)"])
    st.caption("Confusion Matrix")
    st.dataframe(cm)

with col2:
    st.subheader("Accuracy vs #Weak Learners (T)")
    fig, ax = plt.subplots()
    ax.plot(range(1, n_estimators + 1), acc_list, marker="o")
    ax.set_xlabel("Number of weak learners (T)")
    ax.set_ylabel("Test Accuracy")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

st.markdown("---")
st.markdown("**Notes**")
st.write(
    "- (1) 가중치 초기화 → (2) 결정 스텀프 학습 → (3) 오분류율/α 계산 → "
    "(4) 샘플 가중치 업데이트 → (5) 최종 결합 → (6) 평가 흐름으로 구현했습니다.\n"
    "- 알고리즘 로직은 NumPy만 사용해 직접 작성했고, 데이터 로드만 scikit-learn을 활용했습니다.\n"
    "- 사이드바에서 T(약한 분류기 개수)를 조절하면 즉시 재학습합니다.\n"
)
