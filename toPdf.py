import re
import subprocess
from pathlib import Path
import os

# ---------- CONFIG ----------
input_md = Path("README.md")
output_md = Path("README_rendered.md")
output_pdf = Path("example.pdf")

# Full path to mmdc (from your system)
MMDC_PATH = "/Users/antonio/.nvm/versions/node/v24.4.0/bin/mmdc"

# LaTeX main font (system-installed font on macOS)
MAIN_FONT = "Arial"
# ----------------------------

# Add mmdc directory to PATH (in case subprocess needs it)
os.environ["PATH"] += os.pathsep + str(Path(MMDC_PATH).parent)

# Read Markdown
with input_md.open("r", encoding="utf-8") as f:
    md_content = f.read()

# Find Mermaid code blocks
mermaid_blocks = re.findall(r"```mermaid(.*?)```", md_content, re.DOTALL)

temp_files = []

for i, block in enumerate(mermaid_blocks):
    print(f"Generating Mermaid chart {i+1}/{len(mermaid_blocks)}")
    mmd_file = Path(f"mermaid_{i}.mmd")
    img_file = Path(f"mermaid_{i}.png")
    temp_files.extend([mmd_file, img_file])

    # Save Mermaid code
    mmd_file.write_text(block.strip(), encoding="utf-8")

    # Render PNG using Mermaid CLI
    subprocess.run([MMDC_PATH, "-i", str(mmd_file), "-o", str(img_file)], check=True)

    # Replace Mermaid block with image in Markdown
    md_content = md_content.replace(f"```mermaid{block}```", f"![Diagram](./{img_file})")

# Save modified Markdown
output_md.write_text(md_content, encoding="utf-8")
temp_files.append(output_md)  # mark for cleanup

# Generate PDF with Pandoc + XeLaTeX
try:
    subprocess.run([
        "pandoc",
        str(output_md),
        "-o", str(output_pdf),
        "--pdf-engine=xelatex",
        "-V", f"mainfont={MAIN_FONT}"
    ], check=True)
    print("PDF generated:", output_pdf)
finally:
    # Cleanup temporary files
    for f in temp_files:
        if f.exists():
            f.unlink()
    print("Temporary files cleaned up.")
