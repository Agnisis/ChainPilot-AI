import re
from pathlib import Path

target_dir = Path(r"c:\Users\agnis\OneDrive\Desktop\My Workspace\SCMAi\New_Final_Reports")
md_file = target_dir / "VIT_FINAL_MSc_DS_REPORT.md"
tex_file = target_dir / "VIT_FINAL_MSc_DS_REPORT.tex"

with open(md_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

latex_preamble = r"""\documentclass[12pt, a4paper]{report}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{tabularx}
\usepackage{array}
\usepackage{hyperref}
\usepackage{geometry}
\usepackage{float}
\geometry{a4paper, left=1.5in, right=1in, top=1in, bottom=1in}
\usepackage{setspace}
\onehalfspacing
\usepackage{titlesec}
\titleformat{\chapter}[display]
  {\normalfont\huge\bfseries\centering}{\chaptertitlename\ \thechapter}{10pt}{\Huge}
\titlespacing*{\chapter}{0pt}{-30pt}{20pt}

\begin{document}

\begin{titlepage}
    \centering
    \vspace*{1in}
    {\LARGE \textbf{AI-Powered Supply Chain Intelligence Platform}}\par
    \vspace{1.5in}
    {\large Submitted in partial fulfillment of the requirements for the degree of}\par
    \vspace{0.2in}
    {\Large \textbf{Master of Data Science (Online)}}\par
    \vspace{1in}
    {\large by}\par
    \vspace{0.2in}
    {\Large \textbf{[Learner Name]}}\par
    {\large \textbf{[Learner Reg. No]}}\par
    \vspace{1in}
    {\large Under the guidance of \textbf{[Guide Name]}}\par
    \vspace{1in}
    {\large \textbf{June 2026}}\par
    {\large \textbf{VIT Online Learning Program}}\par
\end{titlepage}

\pagenumbering{roman}

\chapter*{DECLARATION}
\addcontentsline{toc}{chapter}{DECLARATION}
I, [Learner Name], with register number [Learner Reg. No] hereby declare that the project report entitled ``AI-Powered Supply Chain Intelligence Platform'' submitted by me to M.Sc., Data Science VIT Online learning program, Vellore, in partial fulfilment of the requirement for the award of the degree of Master of Data Science is a Bonafide work carried out by me under the supervision of Prof. [Guide Name], [Designation], [Department], [School], Vellore Institute of Technology, Vellore - 632 014.

I further declare that the work reported in this project has not been submitted and will not be submitted, either in part or in full, for the award of any other degree or diploma in this institute or any other Institute or University.

\vspace{0.5in}
\noindent Place: VIT VELLORE \newline
Date: [Date] \hfill \textbf{[LEARNER NAME]} \newline
\null \hfill Signature of the Candidate

\chapter*{CERTIFICATE}
\addcontentsline{toc}{chapter}{CERTIFICATE}
This is to certify that the project work entitled ``AI-Powered Supply Chain Intelligence Platform'' submitted by [Learner Name] with registration number [Learner Reg. No], to VIT Vellore, in partial fulfilment of the requirement for the award of the degree of Master of Data Science, is a bona fide work carried out by him/her under my supervision. The project fulfils the requirements as per VIT Vellore regulations and, in my opinion, meets the necessary standards for submission. The contents of this report have not been submitted and will not be submitted either in part or in full, for the award of any other degree or diploma in this Institute or any other Institute or University.

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

I would like to express my special gratitude and thanks to my guide [Guide Name], [Designation], [School], whose esteemed guidance and immense support encouraged me to complete the project successfully.

My sincere thanks to Honourable Chancellor, Dr. G. VISWANATHAN; esteemed Vice-Presidents; respected Vice Chancellor, Dr. V. S. KANCHANA BHAASKARAN of this prestigious VIT, Vellore, for providing me an excellent world-class academic environment and facilities for pursuing my online M.Sc. Data Science Program.

My sincere gratitude lies to the Director, Dr. RHYMEND UTHARIARAJ VITOL, and the Head of the Department, online M.Sc. Data Science, VITOL, Prof. Sri Rama Vara Prasad Bhuvanagiri, for providing me with an opportunity to do my project work at VIT, Vellore.

I also thank all the faculty members of the VITOL, Department of Mathematics and the faculty of other Departments of the VIT, as well as the non-teaching staff, for giving me the courage and strength that I needed to achieve my goals.

My special thanks to my friends for their timely help and suggestions rendered for the successful completion of this project.

This acknowledgement would be incomplete without expressing my whole-hearted thanks to my parents for their continuous support and guidance in all walks of my life.

\vspace{0.5in}
\noindent \textbf{[LEARNER NAME]}

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

body_start_idx = 0
for i, line in enumerate(lines):
    if line.startswith("# CHAPTER 1"):
        body_start_idx = i
        break

lines = lines[body_start_idx:]

def process_inline(text):
    math_blocks = []
    
    def block_repl(match):
        math_blocks.append(match.group(0))
        return f"xXMATHBLOCK{len(math_blocks)-1}Xx"
    text = re.sub(r'\$\$.*?\$\$', block_repl, text)
    
    def inline_repl(match):
        math_blocks.append(match.group(0))
        return f"xXMATHINLINE{len(math_blocks)-1}Xx"
    text = re.sub(r'\$.*?\$', inline_repl, text)
    
    # Escape special characters
    text = text.replace("%", r"\%")
    text = text.replace("&", r"\&")
    text = text.replace("_", r"\_")
    
    # Fix broken brackets
    text = text.replace("<<", "[").replace(">>", "]")
    # Bold
    text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', text)
    # Italics
    text = re.sub(r'\*(.*?)\*', r'\\textit{\1}', text)
    # Inline code (escape properly for LaTeX)
    text = re.sub(r'`(.*?)`', r'\\texttt{\1}', text)
    
    # Restore math blocks
    for i, block in enumerate(math_blocks):
        # We process the math block to remove the $ $ so we can wrap it natively in latex later
        # Actually, for inline math $...$ we can just leave it as $...$
        # Let's restore exactly what was matched
        text = text.replace(f"xXMATHBLOCK{i}Xx", block)
        text = text.replace(f"xXMATHINLINE{i}Xx", block)
        
    return text

tex_lines = []
in_table = False
table_cols = 0
in_code_block = False

image_mapping = {
    "Figure 5.4": "Fig1_Model_RMSE_Comparison.png",
    "Figure 5.5": "Fig2_Actual_vs_Predicted.png",
    "Figure 5.6": "Fig3_Isolation_Forest_Anomalies.png"
}

i = 0
while i < len(lines):
    line = lines[i].rstrip()
    
    # Handle Code Blocks first
    if line.startswith("```"):
        if not in_code_block:
            tex_lines.append(r"\begin{verbatim}")
            in_code_block = True
        else:
            tex_lines.append(r"\end{verbatim}")
            in_code_block = False
        i += 1
        continue
        
    if in_code_block:
        tex_lines.append(line)
        i += 1
        continue
        
    if not line:
        if in_table:
            tex_lines.append(r"\end{tabularx}")
            tex_lines.append(r"\end{table}")
            in_table = False
        tex_lines.append("")
        i += 1
        continue
    
    # Handle Markdown Tables
    if line.startswith("|"):
        processed_line = process_inline(line)
        if not in_table:
            tex_lines.append(r"\begin{table}[H]")
            tex_lines.append(r"\centering")
            
            header_cells = [c.strip().replace(r"\_", " ") for c in processed_line.strip("|").split("|") if c.strip()]
            caption_text = " - ".join(header_cells)
            if len(caption_text) > 80:
                caption_text = caption_text[:77] + "..."
            tex_lines.append(f"\\caption{{{caption_text}}}")
            tex_lines.append(r"\vspace{0.1in}")
            
            cols = processed_line.strip("|").count("|") + 1
            col_format = "|".join([r">{\raggedright\arraybackslash}X"] * cols)
            tex_lines.append(r"\begin{tabularx}{\textwidth}{|" + col_format + r"|}")
            tex_lines.append(r"\hline")
            in_table = True
            table_cols = cols
        
        if "---" in processed_line:
            tex_lines.append(r"\hline")
            i += 1
            continue
        
        row_data = processed_line.strip("|").split("|")
        row_data = [cell.strip() for cell in row_data]
        if len(row_data) > table_cols:
            row_data = row_data[:table_cols]
        elif len(row_data) < table_cols:
            row_data.extend([""] * (table_cols - len(row_data)))
            
        tex_lines.append(" & ".join(row_data) + r" \\ \hline")
        i += 1
        continue
        
    # Handle Native Markdown Images
    img_match = re.match(r'^!\[(.*?)\]\((.*?)\)$', line.strip())
    if img_match:
        caption = img_match.group(1)
        img_path = img_match.group(2)
        # Escape caption for LaTeX
        clean_caption = process_inline(caption.replace("#", r"\#"))
        tex_lines.append(r"\begin{figure}[H]")
        tex_lines.append(r"\centering")
        tex_lines.append(f"\\includegraphics[width=0.85\\textwidth]{{{img_path}}}")
        tex_lines.append(f"\\caption{{{clean_caption}}}")
        tex_lines.append(r"\end{figure}")
        i += 1
        continue

    # Escape # in normal text
    if line.startswith("#"):
        processed_line = process_inline(line)
    else:
        line_clean = line.replace("#", r"\#")
        processed_line = process_inline(line_clean)
        
    # Math Blocks
    if processed_line.startswith("$$") and processed_line.endswith("$$") and len(processed_line) > 4:
        math_content = processed_line[2:-2].strip()
        tex_lines.append(r"\begin{equation}")
        tex_lines.append(math_content)
        tex_lines.append(r"\end{equation}")
        i += 1
        continue
        
    # Math inline is already left as $...$ by process_inline, so we do nothing extra here.
    
    # Page Breaks
    if "<div style='page-break-after: always;'></div>" in processed_line:
        tex_lines.append(r"\clearpage")
        i += 1
        continue
        
    # Headers
    if line.startswith("# CHAPTER"):
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i < len(lines):
            title = lines[i].replace("## ", "").strip()
            tex_lines.append(f"\\chapter{{{process_inline(title)}}}")
        i += 1
        continue
        
    if line.startswith("### "):
        tex_lines.append(f"\\subsection{{{process_inline(line[4:])}}}")
        i += 1
        continue
        
    if line.startswith("## ") and not "TABLE OF CONTENTS" in line and not "LIST OF " in line:
        tex_lines.append(f"\\section{{{process_inline(line[3:])}}}")
        i += 1
        continue
        
    # Lists
    if processed_line.startswith("* ") or processed_line.startswith("- "):
        tex_lines.append(r"\begin{itemize}")
        tex_lines.append(f"\\item {processed_line[2:]}")
        i += 1
        while i < len(lines) and (lines[i].startswith("* ") or lines[i].startswith("- ")):
            clean_l = lines[i].rstrip().replace("#", r"\#")
            tex_lines.append(f"\\item {process_inline(clean_l[2:])}")
            i += 1
        tex_lines.append(r"\end{itemize}")
        continue
        
    # Numbered Lists
    if re.match(r'^\d+\.\s', processed_line):
        tex_lines.append(r"\begin{enumerate}")
        tex_lines.append(f"\\item {re.sub(r'^\\d+\\.\\s', '', processed_line)}")
        i += 1
        while i < len(lines) and re.match(r'^\d+\.\s', lines[i]):
            clean_l = lines[i].rstrip().replace("#", r"\#")
            tex_lines.append(f"\\item {process_inline(re.sub(r'^\\d+\\.\\s', '', clean_l))}")
            i += 1
        tex_lines.append(r"\end{enumerate}")
        continue
        
    # Default paragraph text
    if not processed_line.startswith("#") and not processed_line.startswith(">") and not line.startswith("!["):
        tex_lines.append(processed_line)
        
    i += 1

if in_table:
    tex_lines.append(r"\end{tabularx}")
    tex_lines.append(r"\end{table}")

with open(tex_file, "w", encoding="utf-8") as f:
    f.write(latex_preamble + "\n".join(tex_lines) + "\n\\end{document}\n")

print(f"Flawless LaTeX successfully generated at {tex_file.name}")
