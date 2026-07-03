import re
import base64
from pathlib import Path

working_dir = Path(r"c:\Users\agnis\OneDrive\Desktop\My Workspace\SCMAi\Artifacts&Reports")

print("Starting extraction of charts from Jupyter HTML...")

html_file = working_dir / "AI_Powered_Supply_Chain_Intelligence_Platform.html"
html_content = ""
try:
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
except UnicodeDecodeError:
    with open(html_file, 'r', encoding='latin-1') as f:
        html_content = f.read()

# Find all base64 png images in the HTML
pattern = re.compile(r'data:image/png;base64,([A-Za-z0-9+/=]+)')
matches = pattern.findall(html_content)

print(f"Found {len(matches)} charts in the HTML file.")

for idx, match in enumerate(matches):
    img_data = base64.b64decode(match)
    img_path = working_dir / f"Jupyter_Chart_{idx+1}.png"
    with open(img_path, 'wb') as f:
        f.write(img_data)
    print(f"Saved {img_path.name}")

print("\nCompiling FINAL_REPORT-CHAINPILOT_AI.md...")

chapters = [
    "Thesis_Ch1_Introduction.md",
    "Thesis_Ch2_Literature_Review.md",
    "Thesis_Ch3_Machine_Learning_Theory.md",
    "Thesis_Ch4_Deep_Learning_Theory.md",
    "Thesis_Ch5_Web_Application_Architecture.md",
    "Thesis_Ch6_Results_and_Conclusion.md"
]

final_report_content = "# 🎓 FINAL REPORT: CHAINPILOT AI \n\n"
final_report_content += "## AI-Powered Supply Chain Intelligence Platform\n\n"
final_report_content += "---\n\n"

for ch_name in chapters:
    ch_path = working_dir / ch_name
    if ch_path.exists():
        with open(ch_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Inject Jupyter charts into Chapter 3 (Machine Learning Theory)
            if ch_name == "Thesis_Ch3_Machine_Learning_Theory.md":
                if len(matches) >= 1:
                    content += "\n\n### Exploratory Data Analysis Visualizations\n\n"
                    content += "![Jupyter EDA Chart 1](./Jupyter_Chart_1.png)\n\n"
                if len(matches) >= 2:
                    content += "![Jupyter EDA Chart 2](./Jupyter_Chart_2.png)\n\n"
                if len(matches) >= 3:
                    content += "![Jupyter EDA Chart 3](./Jupyter_Chart_3.png)\n\n"
            
            final_report_content += content + "\n\n<div style='page-break-after: always;'></div>\n\n"

final_report_path = working_dir / "FINAL_REPORT-CHAINPILOT_AI.md"
with open(final_report_path, 'w', encoding='utf-8') as f:
    f.write(final_report_content)

print(f"✅ Master thesis successfully compiled to {final_report_path.name}!")
