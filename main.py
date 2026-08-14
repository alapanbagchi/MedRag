from pathlib import Path
from plugins.parsers.pmc import PMCParser  # Or JATSMarkdownParser, depending on what you named the class


def main():
    parser = PMCParser()
    source = Path("data/raw/cardiology2-25/PMC7616479.2.xml")

    # parse() already returns the fully formatted Markdown string
    # complete with the --- YAML frontmatter --- and the article body!
    markdown_content = parser.parse(source)

    output_file = source.with_suffix(".md")

    # Just write the string directly to the file
    output_file.write_text(markdown_content, encoding="utf-8")

    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()