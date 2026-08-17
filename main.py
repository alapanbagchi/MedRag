# main.py
from pathlib import Path
import json
import yaml
from core.registry import PluginRegistry
from plugins.parsers.pmc import PMCParser
from plugins.builders.ast_builder import MarkdownASTBuilder
from plugins.chunkers.ast_chunker import ASTChunker


def extract_yaml_frontmatter(markdown_text: str) -> tuple[dict, str]:
    if markdown_text.startswith("---"):
        end_idx = markdown_text.find("---", 3)
        if end_idx != -1:
            yaml_str = markdown_text[3:end_idx].strip()
            body = markdown_text[end_idx + 3:].strip()
            try:
                meta = yaml.safe_load(yaml_str) or {}
                return meta, body
            except yaml.YAMLError as e:
                print(f"Warning: Could not parse YAML frontmatter: {e}")
                return {}, body
    return {}, markdown_text


def main():
    registry = PluginRegistry()
    registry.register("pmc_parser", PMCParser())
    registry.register("ast_builder", MarkdownASTBuilder())
    registry.register("ast_chunker", ASTChunker())

    xml_path = Path("data/raw/cardiology/PMC7616479.2.xml")

    # Step A: Parse XML -> Markdown
    full_markdown = registry.get("pmc_parser").parse(xml_path)

    # Step B: Extract Metadata & Body
    metadata, markdown_body = extract_yaml_frontmatter(full_markdown)

    print("=" * 50)
    print("EXTRACTED METADATA:")
    print(f"PMC ID: {metadata.get('article_meta', {}).get('article_ids', {}).get('pmcid')}")
    print(f"Title:  {metadata.get('article_meta', {}).get('title')}")
    print("=" * 50 + "\n")

    # Step C: Build AST
    document_ast = registry.get("ast_builder").build(markdown_body, metadata)

    # Step D: Chunk the AST
    chunks = registry.get("ast_chunker").chunk(document_ast)

    print(f"Generated {len(chunks)} chunks.\n")

    # Step E: Save chunks to file for inspection
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    pmcid = metadata.get('article_meta', {}).get('article_ids', {}).get('pmcid', 'unknown')
    output_file = output_dir / f"{pmcid}_chunks.json"

    # Convert chunks to serializable dicts
    chunks_data = []
    for i, chunk in enumerate(chunks):
        chunks_data.append({
            "index": i,
            "id": chunk.id,
            "chunk_type": chunk.chunk_type,
            "metadata": chunk.metadata,
            "text": chunk.text
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)

    print(f"Chunks saved to: {output_file.resolve()}")
    print(f"Total chunks: {len(chunks)}")

    # Print summary of chunk types
    type_counts = {}
    for chunk in chunks:
        type_counts[chunk.chunk_type] = type_counts.get(chunk.chunk_type, 0) + 1

    print("\nChunk type breakdown:")
    for chunk_type, count in sorted(type_counts.items()):
        print(f"  {chunk_type}: {count}")


if __name__ == "__main__":
    main()