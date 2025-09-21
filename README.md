<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  }
};
</script>
# CSI300Classifier

## Project Background

**CSI300Classifier** is a project developed as part of the **2025 Big Data and Information Services** course at Tongji University. This project applies machine learning techniques to financial big data, focusing on predicting stock price trends of constituent stocks in the **CSI 300 Index**. By utilizing historical trading data of CSI 300 components, this project aims to predict future price movements through time-series classification models, providing valuable insights for investment decisions and market analysis.

---

## Abstract

**CSI300Classifier** is a deep learning-based financial time-series classification model designed to predict stock price movements in the CSI 300 Index. The project uses a sliding window approach to model multi-dimensional time-series data and implements four representative models: **DLinear** (a trend-seasonal decomposition-based linear model), **LSTM** (Long Short-Term Memory Network), **TTTLinear** (Test-Time Training with a custom TTT encoder), and **RMoK** (Mixture-of-KAN Experts model). The project compares these models in terms of performance, offering an efficient and accurate solution for time-series classification in financial data analysis and investment decision-making.

---

## 1. Task Definition

Given one or more financial time series$\mathbf{x}\_t \in \mathbb{R}^C$, where$C$is the feature dimension, the task is to perform binary classification (up or down) for each time step in a sliding window of length$L$. The model learns a mapping function$f$, defined as:

$$
f: \mathbb{R}^{L \times C} \rightarrow \{0, 1\}^L
$$

Where:

- $\mathbf{x}_t = (x_{t,1}, x_{t,2}, \dots, x_{t,C})$is the input feature vector at time step$t$,
- $\text{close}\_t$is the closing price at time step$t$, and$\text{prev\_close}_t$is the closing price at the previous time step.

The label for each time step is constructed based on whether the closing price has increased or decreased compared to the previous time step:

$$
\text{label}_t =
\begin{cases}
1 & \text{if } \text{close}_t \geq \text{prev\_close}_t \\
0 & \text{if } \text{close}_t < \text{prev\_close}_t
\end{cases}
$$

The task is to predict the movement of the stock price for future time steps, where the output labels are binary,$y_t \in \{0, 1\}$, indicating whether the stock price increased or decreased.

---

## 2. Data Preprocessing and Feature Engineering

### 2.1 Data Preprocessing

- **Data Cleaning and Alignment**: Raw data is read from CSV files containing daily trading data for CSI 300 constituent stocks. The data is cleaned by removing duplicates, filling missing values, and aligning the dates to ensure consistency.
- **Sliding Window Slicing**: A sliding window approach of length `seq_len` is used to slice the time series data, with each window representing a training sample for prediction.

### 2.2 Feature Engineering

1. **Technical Indicators**:

   - **Moving Averages (MA)**: Calculate the `5-day`, `10-day`, `20-day` moving averages for the closing price.
    $$
     \text{MA}_n = \frac{1}{n} \sum_{i=1}^{n} \text{close}_{t-i}
    $$
   - **Return Calculations**: Calculate daily, weekly, and monthly returns based on the previous day's closing price:
    $$
     \text{daily\_return}_t = \frac{\text{close}_t - \text{close}_{t-1}}{\text{close}_{t-1}}
    $$
     Similarly, `weekly_return` (5-day return) and `monthly_return` (20-day return) are computed.

2. **Label Construction**: The label for each time step is determined based on whether the stock price has increased compared to the previous day's closing price:

$$
   \text{label}_t =
   \begin{cases}
   1 & \text{if } \text{close}_t \geq \text{prev\_close}_t \\
   0 & \text{if } \text{close}_t < \text{prev\_close}_t
   \end{cases}
$$

   This label is used for the subsequent binary classification task.

3. **Data Standardization**: All numerical features, such as `prev_close`, `open`, `high`, `low`, and `volume`, are standardized using **MinMaxScaler** to ensure all features are on the same scale, aiding model convergence.

4. **Sliding Window Slicing**: Each sample is constructed from `seq_len` historical data points, with the label corresponding to the price movement for each time step. The final data format is:$N \times L \times S \times F$, where:
   - $N$: Number of samples,
   - $L$: Length of the window (time steps),
   - $S$: Number of stocks,
   - $F$: Number of features.

---

## 3. Model Architecture

### 3.1 **DLinear**

DLinear is based on **trend-seasonal decomposition**, where the time series is decomposed into trend and seasonal components. These components are modeled separately using linear models and then combined before passing through a classification head.

### 3.2 **LSTM**

LSTM is a classical recurrent neural network used for time-series data that can learn long-range dependencies. The model learns temporal features through multiple stacked LSTM layers and then applies a fully connected layer for classification.

### 3.3 **TTTLinear**

TTTLinear is based on **Test-Time Training (TTT)**, a novel sequence modeling approach introduced in the paper "Learning to (Learn at Test Time): RNNs with Expressive Hidden States" by Yu Sun et al. The key idea of TTT is to make the hidden state itself a machine learning model, and the update rule as a step of self-supervised learning. The model's hidden state is updated during training even on test sequences, enabling improved expressiveness and performance over traditional RNNs. TTTLinear specifically uses a linear model for updating the hidden state.

### 3.4 **RMoK**

RMoK adopts a **Mixture-of-KAN Experts** architecture, where multiple expert models, such as **TaylorKAN** and **WaveKAN**, are used to model the time-series data. A gating mechanism is applied to combine the outputs of these expert models, providing a flexible and adaptive approach to time-series forecasting.

---

## 4. Training and Evaluation

### 4.1 **Training Process**

The model is trained using the **Adam optimizer**, and the learning rate is dynamically adjusted using the `lradj` strategy. An **EarlyStopping** mechanism monitors the performance on the validation set and stops training early if the performance does not improve.

### 4.2 **Evaluation Metrics**

The following metrics are used for evaluation:

- **Accuracy**: The proportion of correct predictions.
- **F1-Score**: A balance between precision and recall.
- **Precision**: The proportion of true positives among all predicted positives.
- **Recall**: The proportion of true positives among all actual positives.
- **ROC-AUC**: The area under the receiver operating characteristic curve, used to measure classification performance.

During evaluation, the model outputs the probability of each class for each time step. The probabilities are normalized using **softmax**, and the final predictions are made by taking the **argmax**.

---

## 5. Quick Start

Clone the repository:

```bash
git clone https://github.com/Kian-Chen/CSI300Classifier.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run experiments:

Through scripts for different models:

```bash
bash scripts/DLinear.sh
bash scripts/LSTM.sh
bash scripts/TTTLinear.sh
bash scripts/RMoK.sh
```

Or use the command line manually:

```bash
python run.py \
    --model_id "CSI300" \
    --is_training 1 \
    --model "DLinear" \
    --root_path "./dataset/" \
    --data_path "CSI300.csv" \
    --log_dir "./logs/" \
    --log_name "DLinear.txt" \
    --data "CSI300" \
    --features "M" \
    --seq_len 20 \
    --pred_len 20 \
    --enc_in 13 \
    --dec_in 13 \
    --d_model 32 \
    --itr 1 \
    --train_epochs 100 \
    --batch_size 256 \
    --patience 3 \
    --learning_rate 0.001 \
    --des "Exp" \
    --lradj "type3" \
    --use_multi_scale "false" \
    --small_kernel_merged "false" \
    --gpu 0
```

---

## 6. Dependencies

```bash
pip install -r requirements.txt
```

---

## 7. Project Structure

```
CSI300Classifier
├── data_provider/        # Data loading and preprocessing
├── exp/                  # Training and evaluation
├── layers/               # Custom network layers and encoders
├── models/               # DLinear / LSTM / TTTLinear / RMoK
├── notebooks/            # Preprocessing, feature engineering, and experiments
├── optimizer/            # Optimizer implementations
├── scripts/              # Batch running experiment scripts
├── utils/                # Utility functions
├── run.py                # Main program entry point
└── dataset/              # Dataset files (e.g., CSI300.csv)
```

---

## 8. Contributing

Contributions are welcome! Please submit an issue or suggestion in the repository, and then submit a PR with your improvements.
