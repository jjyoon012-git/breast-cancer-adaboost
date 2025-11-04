## 🧬 breast-cancer-adaboost-from-scratch  
Decision Stump 기반 AdaBoost 구현 및 Streamlit 시각화  

직접 구현한 AdaBoost 알고리즘을 활용하여  
약한 분류기(Weak Learner)가 순차적으로 추가될 때 테스트 정확도의 변화를 관찰한 과제입니다.  
Breast Cancer Wisconsin Dataset을 기반으로, Decision Stump를 반복 결합하여  
강력한 분류기를 형성하는 AdaBoost의 핵심 아이디어를 실험적으로 분석하였습니다.  

---

## 개요  

본 프로젝트는 **AdaBoost (Adaptive Boosting)** 알고리즘을 **NumPy만 사용하여 직접 구현**한 후,  
Streamlit을 활용해 **약한 분류기의 개수(T)** 변화에 따른 **테스트 정확도 변화**를 시각적으로 확인할 수 있도록 제작되었습니다.  

- **Weak Learner:** 깊이가 1인 Decision Stump
- **Dataset:** Breast Cancer Wisconsin (Diagnostic)  
- **Samples:** 569  
- **Features:** 30개의 수치형 특성  
- **Target:** 0 = Benign, 1 = Malignant

---

##  주요 구성 및 구현 과정  

| 단계 | 설명 |
|------|------|
| STEP 1 | 모든 샘플에 대해 동일한 초기 가중치 \(w = 1/N\) 부여 |
| STEP 2 | 가중 오분류율이 최소가 되는 Decision Stump 학습 |
| STEP 3 | 분류기 가중치(α) 계산 및 오분류율 반영 |
| STEP 4 | 샘플 가중치 업데이트 (오분류된 샘플의 가중치 ↑) |
| STEP 5 | 최종 예측 결합 \(F(x) = \sum_t \alpha_t h_t(x)\) |
| STEP 6 | 테스트 데이터로 정확도 평가 및 시각화 |

---

##  성능 분석  

T (약한 분류기 개수)에 따른 테스트 정확도 변화를 아래와 같이 관찰하였습니다.  

| T | Test Accuracy | 분석 요약 |
|---|----------------|------------|
| 1 | 92.11% | 단일 결정 스텀프로 단순한 경계 형성 |
| 5 | 92.98% | 오분류 샘플 가중 반영으로 정확도 상승 |
| 15 | 95.61% | 모델 수렴, 복잡한 결정 경계 형성 |
| 30 | 95.61% | 수렴 완료, 안정적인 정확도 유지 |
| 100 | 96.49% | 미세한 향상, 학습 시간은 급격히 증가 |

**결론적으로**,  
AdaBoost는 소수의 약한 분류기(약 20~30개) 만으로도 빠르게 수렴하며,  
이후에는 정확도 향상이 미세하고 학습 시간이 비효율적으로 증가했습니다.  
이는 AdaBoost의 핵심 아이디어인 “단순한 학습기의 반복적 결합으로 강한 분류기 형성”을 잘 보여줍니다.

---

## Streamlit UI 기능 요약  

- **Sidebar Parameter Control**
  - 약한 분류기 개수 T 조절  
  - Test Set 비율 조정  
  - Random Seed 설정  
- **Main Display**
  - Accuracy vs Weak Learners (T) 그래프  
  - Confusion Matrix 표  
  - 최종 Test Accuracy 출력  
- **데이터 미리보기 옵션**  
  - `st.dataframe(df.head())` 를 통해 데이터셋 구조 확인 가능  

---

## How to Use 

```bash
git clone https://github.com/<your-username>/breast-cancer-adaboost.git
cd breast-cancer-adaboost
pip install -r requirements.txt
streamlit run mldlhw2.py
