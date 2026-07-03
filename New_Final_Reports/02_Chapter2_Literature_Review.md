# CHAPTER 2

## REVIEW OF LITERATURE

### 2.1 Evolution of Supply Chain Analytics

Supply chain management (SCM) has evolved through three distinct generations of analytical capability. The first generation (1990s-2000s) relied on Enterprise Resource Planning (ERP) systems such as SAP and Oracle, which provided deterministic Material Requirements Planning (MRP) calculations based on fixed lead times, static safety stock formulas, and historical average demand. While these systems digitized supply chain records, they offered no predictive intelligence (Christopher, 2016).

The second generation (2010s) introduced Business Intelligence (BI) dashboards powered by tools such as Tableau, Power BI, and Qlik. These platforms enabled descriptive analytics -- visualizing what happened in the past -- but they could not forecast what would happen in the future. Supply chain managers could see that demand dropped last quarter, but they had no algorithmic guidance on what demand would look like next quarter (Ivanov et al., 2019).

The third and current generation leverages Artificial Intelligence (AI) and Machine Learning (ML) to deliver prescriptive analytics: systems that not only predict future demand but also recommend specific actions to optimize inventory, procurement, and logistics. This thesis contributes to this third generation by building a complete, deployable AI platform that spans the full analytics maturity spectrum from descriptive dashboards to prescriptive LLM-powered recommendations.

### 2.2 Statistical Time-Series Methods

The foundational statistical approach to demand forecasting is the Auto-Regressive Integrated Moving Average (ARIMA) model, formalized by Box and Jenkins (1976). ARIMA models decompose a time series into three components: an autoregressive (AR) term that captures the linear dependency of the current observation on previous observations, a differencing (I) term that removes non-stationarity, and a moving average (MA) term that models the dependency between an observation and a residual error from a lagged observation.

The ARIMA model is defined as:

$$\phi(B)(1-B)^d X_t = \theta(B)\epsilon_t$$

Where $\phi(B)$ is the AR polynomial, $\theta(B)$ is the MA polynomial, $B$ is the backshift operator, $d$ is the differencing order, and $\epsilon_t$ is white noise.

The Seasonal ARIMA (SARIMA) extension adds seasonal differencing to handle periodic patterns:

$$\phi(B)\Phi(B^s)(1-B)^d(1-B^s)^D X_t = \theta(B)\Theta(B^s)\epsilon_t$$

While ARIMA and SARIMA remain widely taught in academic curricula, they suffer from critical limitations in modern supply chain contexts: (1) they assume linear relationships between past and future values, (2) they cannot incorporate exogenous variables (such as promotions, weather, or competitor actions) without extension to ARIMAX, (3) they require the time series to be univariate and stationary after differencing, and (4) they scale poorly to datasets with thousands of SKUs, requiring a separate model to be fit for each individual product (Hyndman & Athanasopoulos, 2021).

The `pmdarima` library used in this project implements an automated ARIMA model selection algorithm (Auto-ARIMA) that systematically searches across combinations of $(p, d, q)$ parameters using the Akaike Information Criterion (AIC) to select the optimal model configuration without manual intervention (Smith & Taylor, 2019).

### 2.3 Machine Learning in Demand Forecasting

The limitations of linear statistical models motivated the adoption of non-linear machine learning algorithms for demand forecasting. Two ensemble methods have emerged as dominant in the supply chain domain: Random Forest and Extreme Gradient Boosting (XGBoost).

**Random Forest (Breiman, 2001)** is a bagging ensemble that constructs multiple independent decision trees on bootstrapped subsets of the training data and aggregates their predictions through majority voting (classification) or averaging (regression). The key mathematical innovation is the introduction of feature randomness at each split point, which decorrelates the individual trees and dramatically reduces overfitting. For a forest of $T$ trees, the prediction is:

$$\hat{y} = \frac{1}{T}\sum_{t=1}^{T}h_t(x)$$

Where $h_t(x)$ is the prediction of tree $t$. Random Forest's interpretability through feature importance rankings makes it particularly valuable in supply chain contexts where stakeholders need to understand which operational variables drive costs and demand (Carbonneau et al., 2008).

**XGBoost (Chen & Guestrin, 2016)** is a gradient boosting framework that constructs trees sequentially, with each new tree trained to correct the residual errors of the ensemble built so far. The objective function combines a loss term and a regularization term:

$$\mathcal{L}(\phi) = \sum_{i}l(\hat{y}_i, y_i) + \sum_{k}\Omega(f_k)$$

Where $l$ is a differentiable convex loss function and $\Omega(f_k) = \gamma T + \frac{1}{2}\lambda\|w\|^2$ is the regularization penalty that controls tree complexity. XGBoost has consistently dominated machine learning competitions (Kaggle) and has been adopted by major enterprises including Amazon, Alibaba, and Walmart for demand planning (Chen & Guestrin, 2016).

**Bayesian Hyperparameter Optimization (Optuna)** represents the state-of-the-art approach to model tuning. Traditional grid search and random search are computationally wasteful because they explore the hyperparameter space uniformly, regardless of which regions have proven promising. Optuna (Akiba et al., 2019) implements the Tree-structured Parzen Estimator (TPE) algorithm, which builds a probabilistic model of the objective function and concentrates the search on regions of the hyperparameter space that are most likely to yield improved performance. This Bayesian approach achieves optimal hyperparameters in significantly fewer trials than brute-force methods.

### 2.4 Deep Learning for Sequential Data

While ensemble methods excel at cross-sectional regression (predicting a target from a fixed set of features), they fundamentally cannot model the temporal ordering of observations. A Random Forest treats each row of data as an independent sample, ignoring the sequential relationships that are critical in time-series forecasting.

**Recurrent Neural Networks (RNNs)** were designed specifically to process sequential data by maintaining a hidden state vector $h_t$ that accumulates information from previous time steps:

$$h_t = f(W_{hh}h_{t-1} + W_{xh}x_t + b)$$

However, standard RNNs suffer from the **vanishing gradient problem** (Bengio et al., 1994): during backpropagation through time (BPTT), gradients are multiplied by the weight matrix at each time step, causing them to shrink exponentially. This makes it impossible for the network to learn dependencies beyond approximately 10-20 time steps.

**Long Short-Term Memory (LSTM) networks** (Hochreiter & Schmidhuber, 1997) solve the vanishing gradient problem by introducing a cell state $C_t$ and three gating mechanisms:

**Forget Gate:** Determines what information to discard from the cell state.
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Input Gate:** Determines what new information to store in the cell state.
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**Cell State Update:**
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Output Gate:** Determines what information to output.
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

Where $\sigma$ is the sigmoid activation function and $\odot$ denotes element-wise multiplication.

**Gated Recurrent Units (GRU)** (Cho et al., 2014) offer a simplified alternative that merges the forget and input gates into a single update gate, reducing computational complexity while maintaining comparable performance:

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$
$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

Both LSTM and GRU architectures have demonstrated significant improvements over ARIMA for multi-step-ahead demand forecasting in supply chain contexts, particularly when trained on datasets with strong seasonal patterns and promotional effects (Bandara et al., 2020).

### 2.5 Anomaly Detection in Supply Chains

Supply chain disruptions -- including supplier fraud, transportation delays, quality defects, and demand shocks -- represent a major source of financial loss. Traditional anomaly detection relies on univariate statistical process control (SPC) charts (Shewhart charts, CUSUM) that monitor a single variable against fixed control limits. These methods cannot detect anomalies that manifest only when multiple variables are considered jointly (Chalapathy & Chawla, 2019).

**Isolation Forest** (Liu et al., 2008) is an unsupervised anomaly detection algorithm based on the principle that anomalies are few and different. Instead of profiling normal behavior and then detecting deviations (as in One-Class SVM), Isolation Forest directly isolates anomalies by randomly selecting a feature and a random split value. Anomalous points, being rare and having extreme feature values, require fewer random splits to be isolated from the rest of the data. The anomaly score for a point $x$ is:

$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

Where $E(h(x))$ is the average path length from the root to the point across all isolation trees, and $c(n)$ is the average path length of unsuccessful search in a Binary Search Tree. A score close to 1 indicates a strong anomaly; a score close to 0.5 indicates a normal observation.

### 2.6 Large Language Models and RAG Systems

The release of transformer-based Large Language Models (LLMs) -- including GPT-4 (OpenAI, 2023) and Gemini (Google DeepMind, 2024) -- has created an unprecedented opportunity to bridge the gap between complex quantitative analytics and human-readable strategic recommendations. However, LLMs are prone to hallucination: generating plausible-sounding but factually incorrect information (Ji et al., 2023).

**Retrieval-Augmented Generation (RAG)** (Lewis et al., 2020) addresses this limitation by grounding LLM responses in specific, retrieved context documents. The RAG pipeline consists of three stages:

1. **Document Ingestion:** Enterprise documents (PDFs, SOPs, vendor contracts) are split into chunks and embedded into a high-dimensional vector space using a pre-trained sentence transformer model.

2. **Vector Storage:** The embeddings are stored in a vector database (such as FAISS or ChromaDB) that supports efficient approximate nearest-neighbor search.

3. **Retrieval-Augmented Response:** When a user asks a question, the query is embedded, the most semantically similar document chunks are retrieved, and they are injected into the LLM's prompt as context. The LLM then generates a response that is grounded in the retrieved documents, dramatically reducing hallucination.

This thesis implements a complete RAG pipeline using ChromaDB for vector storage, the `all-MiniLM-L6-v2` sentence transformer for embeddings, and Google Gemini as the generative LLM.

### 2.7 Summary of Research Gaps

The literature review reveals the following critical gaps that this thesis addresses:

| Gap | How ChainPilot AI Addresses It |
|-----|-------------------------------|
| Most ML supply chain research stays in Jupyter Notebooks | Full-stack FastAPI + React deployment |
| Models are typically tested on a single dataset | Validated across 4 diverse real-world domains |
| Ensemble and DL models lack interpretability for executives | RAG + Gemini LLM translates metrics to strategy |
| Anomaly detection uses univariate methods | Multivariate Isolation Forest on high-dimensional data |
| Hyperparameter tuning uses brute-force grid search | Optuna Bayesian optimization (TPE algorithm) |
