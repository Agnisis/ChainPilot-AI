# ChainPilot AI: An AI-Powered Multi-Domain Supply Chain Intelligence Platform Using Deep Learning, Ensemble Methods, and Retrieval-Augmented Generation

Submitted in partial fulfillment of the requirements for the degree of

**Master of Data Science (Online)**

by

**<<Learner Name>>**

**<<Learner Reg. No>>**

Under the guidance of **<<Guide Name>>**

**June 2026**

**VIT Online Learning Program**

---

## DECLARATION

I, <<Learner Name>>, with register number <<Learner Reg. No>> hereby declare that the project report entitled "ChainPilot AI: An AI-Powered Multi-Domain Supply Chain Intelligence Platform Using Deep Learning, Ensemble Methods, and Retrieval-Augmented Generation" submitted by me to M.Sc., Data Science VIT Online learning program, Vellore, in partial fulfilment of the requirement for the award of the degree of Master of Data Science is a Bonafide work carried out by me under the supervision of Prof. <<Guide Name>>, <<Designation>>, <<Department>>, <<School>>, Vellore Institute of Technology, Vellore - 632 014.

I further declare that the work reported in this project has not been submitted and will not be submitted, either in part or in full, for the award of any other degree or diploma in this institute or any other Institute or University.

Place: VIT VELLORE
Date: <<Date>>

<<LEARNER NAME>>
Signature of the Candidate

---

## CERTIFICATE

This is to certify that the project work entitled "ChainPilot AI: An AI-Powered Multi-Domain Supply Chain Intelligence Platform Using Deep Learning, Ensemble Methods, and Retrieval-Augmented Generation" submitted by <<Learner Name>> with registration number <<Learner Reg. No>>, to VIT Vellore, in partial fulfilment of the requirement for the award of the degree of Master of Data Science, is a bona fide work carried out by him/her under my supervision. The project fulfils the requirements as per VIT Vellore regulations and, in my opinion, meets the necessary standards for submission. The contents of this report have not been submitted and will not be submitted either in part or in full, for the award of any other degree or diploma in this Institute or any other Institute or University.

Place: VIT VELLORE
Date:

Guide Name & Signature
Examiner 1: HOD Online M.Sc. DS
Examiner 2: Director, VITOL

---

## ACKNOWLEDGEMENT

At the outset, I thank the Almighty God for His blessings for granting me the knowledge and right aptitude to successfully complete my project work.

I would like to express my special gratitude and thanks to my guide <<Guide Name>>, <<Designation>>, <<School>>, whose esteemed guidance and immense support encouraged me to complete the project successfully.

My sincere thanks to Honourable Chancellor, Dr. G. VISWANATHAN; esteemed Vice-Presidents; respected Vice Chancellor, Dr. V. S. KANCHANA BHAASKARAN of this prestigious VIT, Vellore, for providing me an excellent world-class academic environment and facilities for pursuing my online M.Sc. Data Science Program.

My sincere gratitude lies to the Director, Dr. RHYMEND UTHARIARAJ VITOL, and the Head of the Department, online M.Sc. Data Science, VITOL, Prof. Sri Rama Vara Prasad Bhuvanagiri, for providing me with an opportunity to do my project work at VIT, Vellore.

I also thank all the faculty members of the VITOL, Department of Mathematics and the faculty of other Departments of the VIT, as well as the non-teaching staff, for giving me the courage and strength that I needed to achieve my goals.

My special thanks to my friends for their timely help and suggestions rendered for the successful completion of this project.

This acknowledgement would be incomplete without expressing my whole-hearted thanks to my parents for their continuous support and guidance in all walks of my life.

<<LEARNER NAME>>

---

## ABSTRACT

This thesis presents **ChainPilot AI**, a comprehensive, AI-powered, multi-domain supply chain intelligence platform that bridges the gap between isolated data science research and real-world enterprise deployment. The platform integrates six advanced predictive and analytical algorithms -- Auto-ARIMA, Random Forest, XGBoost with Optuna Bayesian optimization, PyTorch Long Short-Term Memory (LSTM) networks, PyTorch Gated Recurrent Units (GRU), and Isolation Forest anomaly detection -- into a unified, production-grade software architecture built with FastAPI (Python backend) and React 18 (JavaScript frontend).

To demonstrate domain independence and commercial scalability, the system was validated across four massive, real-world benchmark datasets spanning distinct business verticals: (1) M5 Forecasting Accuracy (Walmart hierarchical retail demand), (2) Rossmann Store Sales (European retail promotion forecasting), (3) DataCo Smart Supply Chain (global logistics and shipping analytics), and (4) Brazilian E-Commerce Olist (end-to-end e-commerce fulfillment and freight analysis). The architecture employs walk-forward cross-validation to prevent data leakage, SHAP-based feature importance analysis for model interpretability, and a Retrieval-Augmented Generation (RAG) pipeline powered by Google Gemini and FAISS vector databases to translate complex quantitative outputs into natural-language executive recommendations.

Empirical results demonstrate that Optuna-tuned XGBoost and PyTorch LSTM architectures consistently outperform traditional statistical methods (Auto-ARIMA) across all four domains, while the Isolation Forest algorithm successfully detects multivariate supply chain anomalies that standard univariate monitoring would miss. The web application provides interactive Chart.js visualizations, dynamic KPI dashboards, and a conversational AI assistant, proving that state-of-the-art predictive analytics can be deployed as an intuitive, real-time Software-as-a-Service (SaaS) solution.

**Keywords:** Supply Chain Management, Deep Learning, LSTM, XGBoost, Optuna, Anomaly Detection, Retrieval-Augmented Generation, FastAPI, React, FAISS, Demand Forecasting, PyTorch

---

## TABLE OF CONTENTS

| Section | Title | Page |
|---------|-------|------|
| | Declaration | i |
| | Certificate | ii |
| | Acknowledgement | iii |
| | Abstract | iv |
| | Table of Contents | v |
| | List of Figures | vi |
| | List of Tables | vii |
| **1** | **INTRODUCTION** | **1** |
| 1.1 | Background and Motivation | 1 |
| 1.2 | Problem Statement | 2 |
| 1.3 | Research Objectives | 3 |
| 1.4 | Research Questions | 4 |
| 1.5 | Scope and Multi-Domain Data Sources | 4 |
| 1.6 | Organization of the Report | 6 |
| **2** | **REVIEW OF LITERATURE** | **7** |
| 2.1 | Evolution of Supply Chain Analytics | 7 |
| 2.2 | Statistical Time-Series Methods | 8 |
| 2.3 | Machine Learning in Demand Forecasting | 9 |
| 2.4 | Deep Learning for Sequential Data | 10 |
| 2.5 | Anomaly Detection in Supply Chains | 11 |
| 2.6 | Large Language Models and RAG Systems | 12 |
| 2.7 | Summary of Research Gaps | 12 |
| **3** | **SYSTEM ARCHITECTURE AND PROPOSED METHODOLOGY** | **13** |
| 3.1 | High-Level System Architecture | 13 |
| 3.2 | Backend Architecture (FastAPI) | 14 |
| 3.3 | Frontend Architecture (React 18) | 16 |
| 3.4 | Data Ingestion and Preprocessing Pipeline | 17 |
| 3.5 | Multi-Domain Dataset Profiles | 18 |
| 3.6 | RAG Pipeline: FAISS + Google Gemini | 20 |
| **4** | **IMPLEMENTATION DETAILS** | **21** |
| 4.1 | Data Preprocessing and Feature Engineering | 21 |
| 4.2 | Auto-ARIMA Implementation | 23 |
| 4.3 | Random Forest Regressor | 24 |
| 4.4 | XGBoost with Optuna Bayesian Optimization | 25 |
| 4.5 | PyTorch LSTM Forecaster | 27 |
| 4.6 | PyTorch GRU Forecaster | 28 |
| 4.7 | Isolation Forest Anomaly Detection | 29 |
| 4.8 | Walk-Forward Cross-Validation | 30 |
| **5** | **RESULTS AND DISCUSSION** | **31** |
| 5.1 | Exploratory Data Analysis | 31 |
| 5.2 | Model Performance Comparison | 33 |
| 5.3 | Deep Learning Convergence Analysis | 35 |
| 5.4 | Anomaly Detection Outcomes | 36 |
| 5.5 | Web Application Dashboard | 37 |
| **6** | **CONCLUSION AND FUTURE SCOPE** | **38** |
| 6.1 | Summary of Contributions | 38 |
| 6.2 | Limitations | 39 |
| 6.3 | Future Scope | 39 |
| | References | 40 |

---

## LIST OF FIGURES

| Figure No. | Title |
|------------|-------|
| Figure 1.1 | High-Level Architecture of ChainPilot AI |
| Figure 3.1 | System Architecture Diagram |
| Figure 3.2 | FastAPI Backend Request Flow |
| Figure 3.3 | React 18 Component Hierarchy |
| Figure 3.4 | RAG Pipeline: Document Ingestion to LLM Response |
| Figure 4.1 | LSTM Cell Architecture with Forget, Input, and Output Gates |
| Figure 4.2 | GRU Cell Architecture |
| Figure 4.3 | Optuna Bayesian Optimization Search Space |
| Figure 5.1 | Demand Trend Analysis (Jupyter Notebook EDA) |
| Figure 5.2 | Feature Correlation Heatmap (Jupyter Notebook EDA) |
| Figure 5.3 | Data Distribution Histogram (Jupyter Notebook EDA) |
| Figure 5.4 | Model RMSE Comparison Bar Chart |
| Figure 5.5 | PyTorch LSTM: Actual vs Predicted Demand |
| Figure 5.6 | Isolation Forest: Multivariate Anomaly Scatter Plot |
| Figure 5.7 | Web Application Dashboard Screenshot |

## LIST OF TABLES

| Table No. | Title |
|-----------|-------|
| Table 1.1 | Multi-Domain Dataset Summary |
| Table 3.1 | Technology Stack |
| Table 4.1 | Optuna Hyperparameter Search Space for XGBoost |
| Table 4.2 | PyTorch LSTM/GRU Model Configuration |
| Table 5.1 | Model Performance Metrics Across Domains |
| Table 5.2 | Anomaly Detection Results |
