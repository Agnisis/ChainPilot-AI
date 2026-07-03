import re
from pathlib import Path

target_dir = Path(r"c:\Users\agnis\OneDrive\Desktop\My Workspace\SCMAi\New_Final_Reports")
md_file = target_dir / "VIT_FINAL_MSc_DS_REPORT.md"
tex_file = target_dir / "VIT_FINAL_MSc_DS_REPORT.tex"

with open(md_file, "r", encoding="utf-8") as f:
    md_content = f.read()

# LaTeX Preamble for a Master's Thesis
latex_preamble = r"""\documentclass[12pt, a4paper]{report}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{a4paper, left=1.5in, right=1in, top=1in, bottom=1in}
\usepackage{setspace}
\onehalfspacing
\usepackage{titlesec}
\titleformat{\chapter}[display]
  {\normalfont\huge\bfseries\centering}{\chaptertitlename\ \thechapter}{20pt}{\Huge}

\begin{document}

\begin{titlepage}
    \centering
    \vspace*{1in}
    {\LARGE \textbf{ChainPilot AI: An AI-Powered Multi-Domain Supply Chain Intelligence Platform Using Deep Learning, Ensemble Methods, and Retrieval-Augmented Generation}}\par
    \vspace{1.5in}
    {\large Submitted in partial fulfillment of the requirements for the degree of}\par
    \vspace{0.2in}
    {\Large \textbf{Master of Data Science (Online)}}\par
    \vspace{1in}
    {\large by}\par
    \vspace{0.2in}
    {\Large \textbf{<<Learner Name>>}}\par
    {\large \textbf{<<Learner Reg. No>>}}\par
    \vspace{1in}
    {\large Under the guidance of \textbf{<<Guide Name>>}}\par
    \vspace{1in}
    {\large \textbf{June 2026}}\par
    {\large \textbf{VIT Online Learning Program}}\par
\end{titlepage}

\pagenumbering{roman}

\chapter*{DECLARATION}
\addcontentsline{toc}{chapter}{DECLARATION}
I, <<Learner Name>>, with register number <<Learner Reg. No>> hereby declare that the project report entitled ``ChainPilot AI: An AI-Powered Multi-Domain Supply Chain Intelligence Platform Using Deep Learning, Ensemble Methods, and Retrieval-Augmented Generation'' submitted by me to M.Sc., Data Science VIT Online learning program, Vellore, in partial fulfilment of the requirement for the award of the degree of Master of Data Science is a Bonafide work carried out by me under the supervision of Prof. <<Guide Name>>, <<Designation>>, <<Department>>, <<School>>, Vellore Institute of Technology, Vellore - 632 014.

I further declare that the work reported in this project has not been submitted and will not be submitted, either in part or in full, for the award of any other degree or diploma in this institute or any other Institute or University.

\vspace{0.5in}
\noindent Place: VIT VELLORE \newline
Date: <<Date>> \hfill \textbf{<<LEARNER NAME>>} \newline
\null \hfill Signature of the Candidate

\chapter*{CERTIFICATE}
\addcontentsline{toc}{chapter}{CERTIFICATE}
This is to certify that the project work entitled ``ChainPilot AI: An AI-Powered Multi-Domain Supply Chain Intelligence Platform Using Deep Learning, Ensemble Methods, and Retrieval-Augmented Generation'' submitted by <<Learner Name>> with registration number <<Learner Reg. No>>, to VIT Vellore, in partial fulfilment of the requirement for the award of the degree of Master of Data Science, is a bona fide work carried out by him/her under my supervision. The project fulfils the requirements as per VIT Vellore regulations and, in my opinion, meets the necessary standards for submission. The contents of this report have not been submitted and will not be submitted either in part or in full, for the award of any other degree or diploma in this Institute or any other Institute or University.

\vspace{0.5in}
\noindent Place: VIT VELLORE \newline
Date: \newline

\vspace{0.5in}
\noindent Guide Name \& Signature \newline
Examiner 1: HOD Online M.Sc. DS \newline
Examiner 2: Director, VITOL

\chapter*{ACKNOWLEDGEMENT}
\addcontentsline{toc}{chapter}{ACKNOWLEDGEMENT}
At the outset, I thank the Almighty God for His blessings for granting me the knowledge and right aptitude to successfully complete my project work.

I would like to express my special gratitude and thanks to my guide <<Guide Name>>, <<Designation>>, <<School>>, whose esteemed guidance and immense support encouraged me to complete the project successfully.

My sincere thanks to Honourable Chancellor, Dr. G. VISWANATHAN; esteemed Vice-Presidents; respected Vice Chancellor, Dr. V. S. KANCHANA BHAASKARAN of this prestigious VIT, Vellore, for providing me an excellent world-class academic environment and facilities for pursuing my online M.Sc. Data Science Program.

My sincere gratitude lies to the Director, Dr. RHYMEND UTHARIARAJ VITOL, and the Head of the Department, online M.Sc. Data Science, VITOL, Prof. Sri Rama Vara Prasad Bhuvanagiri, for providing me with an opportunity to do my project work at VIT, Vellore.

I also thank all the faculty members of the VITOL, Department of Mathematics and the faculty of other Departments of the VIT, as well as the non-teaching staff, for giving me the courage and strength that I needed to achieve my goals.

My special thanks to my friends for their timely help and suggestions rendered for the successful completion of this project.

This acknowledgement would be incomplete without expressing my whole-hearted thanks to my parents for their continuous support and guidance in all walks of my life.

\vspace{0.5in}
\noindent \textbf{<<LEARNER NAME>>}

\chapter*{ABSTRACT}
\addcontentsline{toc}{chapter}{ABSTRACT}
This thesis presents \textbf{ChainPilot AI}, a comprehensive, AI-powered, multi-domain supply chain intelligence platform that bridges the gap between isolated data science research and real-world enterprise deployment. The platform integrates six advanced predictive and analytical algorithms---Auto-ARIMA, Random Forest, XGBoost with Optuna Bayesian optimization, PyTorch Long Short-Term Memory (LSTM) networks, PyTorch Gated Recurrent Units (GRU), and Isolation Forest anomaly detection---into a unified, production-grade software architecture built with FastAPI (Python backend) and React 18 (JavaScript frontend).

To demonstrate domain independence and commercial scalability, the system was validated across four massive, real-world benchmark datasets spanning distinct business verticals: (1) M5 Forecasting Accuracy (Walmart hierarchical retail demand), (2) Rossmann Store Sales (European retail promotion forecasting), (3) DataCo Smart Supply Chain (global logistics and shipping analytics), and (4) Brazilian E-Commerce Olist (end-to-end e-commerce fulfillment and freight analysis). The architecture employs walk-forward cross-validation to prevent data leakage, SHAP-based feature importance analysis for model interpretability, and a Retrieval-Augmented Generation (RAG) pipeline powered by Google Gemini and FAISS vector databases to translate complex quantitative outputs into natural-language executive recommendations.

Empirical results demonstrate that Optuna-tuned XGBoost and PyTorch LSTM architectures consistently outperform traditional statistical methods (Auto-ARIMA) across all four domains, while the Isolation Forest algorithm successfully detects multivariate supply chain anomalies that standard univariate monitoring would miss. The web application provides interactive Chart.js visualizations, dynamic KPI dashboards, and a conversational AI assistant, proving that state-of-the-art predictive analytics can be deployed as an intuitive, real-time Software-as-a-Service (SaaS) solution.

\vspace{0.2in}
\noindent \textbf{Keywords:} Supply Chain Management, Deep Learning, LSTM, XGBoost, Optuna, Anomaly Detection, Retrieval-Augmented Generation, FastAPI, React, FAISS, Demand Forecasting, PyTorch

\tableofcontents
\listoffigures
\listoftables

\clearpage
\pagenumbering{arabic}

"""

# We don't want to double-process the front matter that we just hardcoded into the preamble.
# So we slice the markdown starting from Chapter 1.
try:
    ch1_start = md_content.index("# CHAPTER 1")
    body_md = md_content[ch1_start:]
except ValueError:
    body_md = md_content

# Write the body to a temp markdown file
temp_md = target_dir / "temp_body.md"
with open(temp_md, "w", encoding="utf-8") as f:
    f.write(body_md)

with open(target_dir / "template.tex", "w", encoding="utf-8") as f:
    f.write(latex_preamble + "\n$body$\n\\end{document}\n")

print("Created LaTeX template and temp markdown body. Ready for Pandoc conversion.")
