from pathlib import Path
from docling.document_converter import DocumentConverter
output_dir = Path("md")
output_dir.mkdir(parents=True, exist_ok=True)



converter = DocumentConverter()
result = converter.convert(source=str(Path("data/raw/hypertension-mesh-terms-or-blood-pressure-mesh-terms/PMC4322514.xml")))

output_file = output_dir / f"test.md"
output_file.write_text(
    result.document.export_to_markdown(),
    encoding="utf-8"
)

print(f"Saved: {output_file}")
