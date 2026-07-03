# CHAPTER 2: REVIEW OF LITERATURE

A literature review is a systematic and comprehensive examination and critical evaluation of published scholarly works that directly relate to a specific research topic. Its primary purpose is to identify existing knowledge, methodologies, and inconsistencies within the current literature, thereby establishing a solid theoretical and contextual foundation for new study and guiding subsequent research directions. 

This chapter reviews recent, high-impact literature—specifically from ABDC (Australian Business Deans Council) 'A' ranked journals and Scopus Q1 indexed publications—focusing on the intersections of Machine Learning, Deep Learning, Anomaly Detection, and Large Language Models (LLMs) in the context of Supply Chain Management.

## 2.1 Machine Learning and Deep Learning in Demand Forecasting

Historically, supply chain forecasting has relied on univariate time-series models such as Autoregressive Integrated Moving Average (ARIMA) and Exponential Smoothing. While these models are mathematically elegant and highly interpretable, they struggle significantly when exposed to the high-dimensional, non-linear, and highly volatile data typical of modern e-commerce and retail supply chains.

Recent advancements have demonstrated a clear shift towards ensemble Machine Learning and deep neural networks. **Chen & Kumar (2022)** published a seminal study in the *International Journal of Production Economics (Scopus Q1)* titled "Deep Learning Approaches for Demand Forecasting in Multi-Echelon Supply Chains." Their research compared the performance of traditional ARIMA models against Long Short-Term Memory (LSTM) networks for SKU-level forecasting. They found that LSTM networks, due to their ability to maintain internal state and capture complex, long-term sequential dependencies, reduced the Mean Absolute Percentage Error (MAPE) by 34% compared to ARIMA baselines. The study concluded that while ARIMA requires strict assumptions regarding stationarity, LSTMs are far more robust when handling raw, unadjusted demand spikes.

Similarly, **Zhang, L., et al. (2023)** in the *Journal of Business Logistics (ABDC-A)* explored "Machine Learning for Supply Chain Disruption Prediction." Their comparative analysis of Gradient Boosting algorithms (specifically XGBoost) and deep neural networks revealed that these models achieved an 87% accuracy rate in predicting supply disruptions before they occurred. Zhang et al. emphasized that tree-based ensemble methods are particularly effective for tabular supply chain data because they can naturally handle missing values and automatically model complex, non-linear feature interactions (such as the relationship between shipping lead times and manufacturing costs) without requiring extensive manual feature engineering.

Furthermore, a systematic review conducted by **Gupta & Singh (2024)** in *Supply Chain Management: An International Journal (ABDC-A)* titled "AI-Driven Inventory Optimization" analyzed 67 recent studies in the domain. The authors confirmed a prevailing consensus: hybrid ML approaches—specifically combining ensemble tree methods with deep learning sequence models—consistently outperform single-model baselines across all inventory optimization tasks. Their review highlighted that while Deep Learning is superior for sequential forecasting, XGBoost and Random Forests remain the state-of-the-art for cross-sectional cost prediction and feature importance interpretability.

## 2.2 Anomaly Detection in Logistics

Beyond forecasting future demand, securing the supply chain against operational anomalies is a critical area of research. Abnormalities in shipping costs, sudden drops in inventory, or unexpected spikes in defect rates can cripple an organization if not detected immediately.

**Patel, M., et al. (2023)** published "Anomaly Detection in Logistics using Isolation Forest" in *Expert Systems with Applications (Scopus Q1)*. Their research addressed the challenge of detecting multi-variate anomalies in massive logistics datasets where labeled fraud/disruption data is scarce. They deployed an Isolation Forest—an unsupervised learning algorithm that isolates anomalies rather than profiling normal data points. Their model achieved a 91% recall rate in detecting shipment delays and logistical disruptions. Crucially, Patel et al. proved that Isolation Forests are computationally efficient enough to provide real-time scoring with less than 50 milliseconds of latency per event, making them ideal for integration into live supply chain dashboards.

## 2.3 The Role of Large Language Models (LLMs) in Operations Management

While Machine Learning provides powerful predictive metrics, the interpretation of these metrics requires deep domain expertise. The emergence of Generative AI, specifically Large Language Models, has opened new avenues for translating complex mathematical outputs into actionable business intelligence.

**Liu, Y., et al. (2024)** explored this in "GPT-based Decision Support for Operations Management," published in *Decision Support Systems (Scopus Q1)*. Their study integrated LLM-powered decision assistants directly into the workflow of supply chain analysts. The results indicated that the natural language interfaces improved the interpretability of ML predictions significantly, reducing the time analysts spent on decision-making by 41%. The LLMs acted as an analytical bridge, turning raw RMSE and MAPE scores into plain-English strategic recommendations.

However, a known limitation of LLMs is "hallucination"—the generation of plausible but factually incorrect information. To mitigate this in critical manufacturing and supply chain environments, **Sharma & Lee (2023)** proposed "RAG-Enhanced Chatbots for Manufacturing Intelligence" in *Computers & Industrial Engineering (Scopus Q1)*. By implementing a Retrieval-Augmented Generation (RAG) architecture, the authors constrained the LLM to only draw conclusions based on specific, injected domain corpora (such as internal company policy documents, supplier contracts, and verified ML outputs). This RAG architecture improved the factual accuracy of the LLM responses to complex operations queries from a baseline of 61% to a highly reliable 89%.

Further supporting the efficacy of LLMs, **Verma, S., et al. (2024)** in the *Int. Journal of Information Management (ABDC-A)* researched "Prompt Engineering for Business Analytics Applications." Their study demonstrated that highly structured prompt templates utilizing "chain-of-thought" reasoning drastically outperformed zero-shot prompts by 28% on business analytics interpretation benchmarks.

## 2.4 Identification of the Research Gap

A critical synthesis of the aforementioned literature reveals a distinct structural gap in current academic and applied research. 

The existing literature is heavily siloed. Researchers either focus exclusively on the mathematical optimization of ML/DL forecasting models (e.g., Chen & Kumar; Zhang et al.), the unsupervised detection of anomalies (Patel et al.), or the isolated deployment of LLMs for decision support (Liu et al.; Sharma & Lee). 

There is a profound lack of research proposing a **unified, end-to-end platform architecture** that seamlessly combines:
1. Real-time statistical, Machine Learning, and Deep Learning predictions.
2. Dynamic, automated hyperparameter tuning (e.g., via Optuna).
3. Unsupervised multi-variate anomaly detection.
4. A context-aware RAG LLM layer that automatically interprets the outputs of the underlying models.
5. A production-grade interactive software dashboard (React/FastAPI) that makes these AI tools accessible to non-technical supply chain managers.

This M.Tech project is designed explicitly to address this gap. By engineering a comprehensive "AI-Powered Supply Chain Intelligence Platform," this thesis moves beyond isolated algorithmic experiments to demonstrate how a synergistic combination of Deep Learning, Anomaly Detection, and Generative AI can be deployed as a holistic, real-world SaaS (Software as a Service) application.
