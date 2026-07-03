import os
from pathlib import Path

target_dir = Path(r"c:\Users\agnis\OneDrive\Desktop\My Workspace\SCMAi\New_Final_Reports")

chapters = [
    "00_Front_Matter.md",
    "01_Chapter1_Introduction.md",
    "02_Chapter2_Literature_Review.md",
    "03_Chapter3_System_Architecture.md",
    "04_Chapter4_Implementation.md",
    "05_Chapter5_Results_Discussion.md",
    "06_Chapter6_Conclusion.md"
]

compiled_content = ""

for ch in chapters:
    ch_path = target_dir / ch
    if ch_path.exists():
        with open(ch_path, "r", encoding="utf-8") as f:
            compiled_content += f.read()
            compiled_content += "\n\n<div style='page-break-after: always;'></div>\n\n"

output_path = target_dir / "VIT_FINAL_MSc_DS_REPORT.md"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(compiled_content)

print(f"Successfully compiled {len(chapters)} chapters into {output_path.name}")
