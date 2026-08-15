"""
Bulk validation script for PMCParser.
Tests every XML file in a directory against the parser and reports failures.
Includes structural integrity, text preservation, and semantic mapping checks.
"""

import sys
import re
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plugins.parsers.pmc import PMCParser


@dataclass
class FileResult:
    file: str
    passed: bool
    checks_run: int
    checks_passed: int
    failures: list


def extract_outer_image_urls(text: str) -> list:
    urls = []
    i = 0
    while i < len(text):
        idx = text.find('![', i)
        if idx == -1: break
        bracket_count = 1
        j = idx + 2
        while j < len(text) and bracket_count > 0:
            if text[j] == '[':
                bracket_count += 1
            elif text[j] == ']':
                bracket_count -= 1
            j += 1
        if j < len(text) and text[j] == '(':
            paren_count = 1
            k = j + 1
            while k < len(text) and paren_count > 0:
                if text[k] == '(':
                    paren_count += 1
                elif text[k] == ')':
                    paren_count -= 1
                k += 1
            url = text[j + 1:k - 1].strip()
            urls.append(url)
            i = k
        else:
            i = j
    return urls


def validate_single_file(xml_path_str: str) -> FileResult:
    xml_path = Path(xml_path_str)
    failures = []
    checks_run = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal checks_run
        checks_run += 1
        if not condition:
            failures.append(f"{name}: {detail}" if detail else name)

    try:
        parser = PMCParser()
        content = parser.parse(xml_path)
    except Exception as e:
        return FileResult(
            file=xml_path.name, passed=False, checks_run=1, checks_passed=0,
            failures=[f"PARSE CRASH: {type(e).__name__}: {str(e)[:100]}"],
        )

    if content.startswith("---\n"):
        end_idx = content.find("\n---", 4)
        md_body = content[end_idx + 4:].strip() if end_idx != -1 else content
    else:
        md_body = content

    try:
        root = etree.parse(str(xml_path), parser=etree.XMLParser(recover=True)).getroot()
    except Exception:
        root = None

    # CHECK 1: Title
    has_title_in_xml = False
    if root is not None:
        for t in root.xpath("//front/article-meta/title-group/article-title"):
            if (t.text and t.text.strip()) or len(t) > 0:
                has_title_in_xml = True
                break

    if has_title_in_xml:
        h1s = re.findall(r"^# .+$", md_body, re.MULTILINE)
        check("one_h1", len(h1s) >= 1, f"found {len(h1s)}")
        if h1s:
            first_line = md_body.split("\n", 1)[0]
            check("title_first", first_line.startswith("# "), f"first line: '{first_line[:50]}'")
    else:
        checks_run += 2

    # CHECK 2: Abstract (Exclude sub-articles)
    has_abstract_in_xml = False
    if root is not None:
        for abs_node in root.xpath(
                "//abstract[not(@abstract-type='graphical') and not(ancestor::sub-article)] | //trans-abstract[not(ancestor::sub-article)]"):
            text_content = "".join(abs_node.itertext()).strip()
            has_other_children = any(child.tag != 'title' for child in abs_node if isinstance(child.tag, str))
            if has_other_children or (text_content and text_content.lower() not in ('abstract', '')):
                has_abstract_in_xml = True
                break

    if has_abstract_in_xml:
        check("abstract_section", "## Abstract" in md_body)
    else:
        checks_run += 1

    # CHECK 3: References (Exclude sub-articles)
    has_refs_in_xml = False
    if root is not None:
        for rl in root.xpath("//ref-list[not(ancestor::sub-article)]"):
            if rl.find(".//ref") is not None:
                has_refs_in_xml = True
                break

    if has_refs_in_xml:
        check("references_section", "## References" in md_body)
    else:
        checks_run += 1

    # CHECK 4: Body sections (Exclude sub-articles)
    has_sections_in_body = False
    if root is not None:
        for sec in root.xpath("//sec[not(ancestor::sub-article)]"):
            title_node = sec.find("title")
            if title_node is not None and ((title_node.text and title_node.text.strip()) or len(title_node) > 0):
                has_sections_in_body = True
                break

    if has_sections_in_body:
        h2s = re.findall(r"^## (.+)$", md_body, re.MULTILINE)
        core_headings = {"Abstract", "Graphical Abstract", "References"}
        body_sections = [h for h in h2s if h.strip() not in core_headings]
        check("body_sections", len(body_sections) > 0)
    else:
        checks_run += 1

    # CHECK 5: Images (Syntactic)
    image_urls = extract_outer_image_urls(md_body)
    relative_images = [u for u in image_urls if not u.startswith("http")]
    check("images_absolute", not relative_images, str(relative_images[:2]))

    # CHECK 6: xrefs
    internal_links = re.findall(r"\[[^\]]*\]\(#([^)]+)\)", md_body)
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', md_body))
    broken_links = [a for a in internal_links if a not in anchors]
    check("xrefs_resolve", not broken_links, str(broken_links[:3]))

    # CHECK 7, 8, 9: References numbering
    if has_refs_in_xml:
        ref_parts = md_body.split("## References\n")
        if len(ref_parts) > 1:
            ref_text = ref_parts[-1]
        else:
            ref_text = md_body

        numbers = [int(n) for n in re.findall(r'<a id="[^"]+"></a>\n(\d+)\. ', ref_text)]
        check("refs_numbered", len(numbers) > 0)
        if numbers:
            is_sequential = all(numbers[i] == i + 1 for i in range(len(numbers)))
            check("refs_sequential", is_sequential, f"found {numbers[:5]}...")
        ref_anchors = re.findall(r'<a id="([^"]+)"></a>\n\d+\. ', ref_text)
        check("ref_anchors", len(ref_anchors) > 0)
    else:
        checks_run += 3


    # CHECK 10: Figure Count (Exclude sub-articles)
    xml_figs = len(root.xpath("//fig[not(ancestor::sub-article)]")) if root is not None else 0
    md_figs = len(re.findall(r'!\[.*?\]\(.*?\)', md_body))
    if xml_figs > 0:
        check("fig_count", md_figs >= xml_figs * 0.8, f"XML: {xml_figs}, MD: {md_figs}")
    else:
        checks_run += 1

    # CHECK 11: Table Count (Exclude sub-articles)
    xml_tables = len(root.xpath("//table-wrap[not(ancestor::sub-article)]")) if root is not None else 0
    md_tables = md_body.count("<table") + md_body.count("|---")
    if xml_tables > 0:
        check("table_count", md_tables >= xml_tables * 0.8, f"XML: {xml_tables}, MD: {md_tables}")
    else:
        checks_run += 1

    # CHECK 12: Formula Count (Exclude sub-articles)
    xml_formulas = len(
        root.xpath("(//disp-formula | //inline-formula)[not(ancestor::sub-article)]")) if root is not None else 0
    if xml_formulas > 0:
        md_disp_formulas = len(re.findall(r'\$\$.*?\$\$', md_body, re.DOTALL))
        single_dollars = md_body.count("$") - (md_disp_formulas * 4)
        md_inline_formulas = max(0, single_dollars // 2)
        md_formulas = md_disp_formulas + md_inline_formulas
        check("formula_count", md_formulas >= xml_formulas * 0.5, f"XML: {xml_formulas}, MD: {md_formulas}")
    else:
        checks_run += 1

    # CHECK 13: Section Count (Exclude sub-articles)
    xml_secs = len(root.xpath("//sec[not(ancestor::sub-article)]")) if root is not None else 0
    if xml_secs > 0:
        core_headings = {"Abstract", "Graphical Abstract", "References"}
        all_md_headings = re.findall(r"^(#{1,6}) (.+)$", md_body, re.MULTILINE)
        md_secs = len([h for h in all_md_headings if h[1].strip() not in core_headings])
        check("section_count", md_secs >= xml_secs * 0.8, f"XML: {xml_secs}, MD: {md_secs}")
    else:
        checks_run += 1

    # CHECK 14: Text Preservation (Word Count)
    body_nodes = root.xpath(".//body[not(ancestor::sub-article)]") if root is not None else []
    body_node = body_nodes[0] if body_nodes else None

    if body_node is not None:
        xml_text = "".join(body_node.itertext())
        xml_words = len(re.findall(r'\w+', xml_text))

        clean_md = re.sub(r'[*#\[\]\(\)_`~>]', ' ', md_body)
        md_words = len(re.findall(r'\w+', clean_md))

        if xml_words > 50:
            preservation_rate = md_words / xml_words
            check("text_preservation", preservation_rate > 0.85, f"Preserved {preservation_rate:.1%} of words")
        else:
            checks_run += 1
    else:
        checks_run += 1

    # CHECK 15: Image Mapping (Semantic, Exclude sub-articles AND supplementary-material graphics)
    xml_hrefs = set()
    if root is not None:
        for g in root.xpath(
                "(//graphic | //inline-graphic)[not(ancestor::sub-article) and not(ancestor::supplementary-material)]"):
            href = g.get("{http://www.w3.org/1999/xlink}href") or g.get("href")
            if href and not href.startswith("#"):
                xml_hrefs.add(href.split('/')[-1])

    md_urls = extract_outer_image_urls(md_body)
    md_filenames = set(url.split('/')[-1].split('?')[0] for url in md_urls if url.startswith("http"))

    if xml_hrefs:
        matched_images = len(xml_hrefs.intersection(md_filenames))
        match_rate = matched_images / len(xml_hrefs)
        check("image_mapping", match_rate > 0.80, f"Matched {match_rate:.1%} of images")
    else:
        checks_run += 1

    # CHECK 16: Supplementary Material Rendering (Exclude sub-articles)
    xml_supps = root.xpath("//supplementary-material[not(ancestor::sub-article)]") if root is not None else []
    if xml_supps:
        found_supp = False
        for supp in xml_supps:
            label = supp.findtext("label")
            media = supp.find(".//media")
            if media is None:
                media = supp.find(".//graphic")
            href = ""
            if media is not None:
                href = media.get("{http://www.w3.org/1999/xlink}href") or media.get("href") or ""

            if label and label.strip() and label.strip() in content:
                found_supp = True
                break
            if href and href.split('/')[-1] in content:
                found_supp = True
                break

        if not found_supp and "supplementary_assets" in content:
            found_supp = True

        check("supp_material_rendered", found_supp, "Supplementary material not found in MD")
    else:
        checks_run += 1
    checks_passed = checks_run - len(failures)
    return FileResult(
        file=xml_path.name, passed=len(failures) == 0,
        checks_run=checks_run, checks_passed=checks_passed, failures=failures,
    )


def write_json_report(results: list, path: Path, stats: dict):
    data = {**stats, "files": [asdict(r) for r in sorted(results, key=lambda x: x.file)]}
    path.write_text(json.dumps(data, indent=2))


def write_text_report(results: list, path: Path, stats: dict):
    lines = [
        "=" * 70, "PMC PARSER VALIDATION REPORT",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", "=" * 70,
        f"Total files: {stats['total']}", f"Passed: {stats['passed']}",
        f"Failed: {stats['failed']}", f"Pass rate: {stats['pass_rate']}%",
        f"Elapsed: {stats['elapsed_seconds']}s", f"Rate: {stats['files_per_second']} files/s", ""
    ]
    failure_counts = {}
    for r in results:
        for f in r.failures:
            key = f.split(":")[0].strip()
            failure_counts[key] = failure_counts.get(key, 0) + 1
    if failure_counts:
        lines.extend(["-" * 70, "FAILURE BREAKDOWN BY TYPE", "-" * 70])
        for check_name, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {check_name:<30} {count:>6} files")
        lines.append("")
    failed = [r for r in results if not r.passed]
    if failed:
        lines.extend(["-" * 70, f"FAILED FILES ({len(failed)})", "-" * 70])
        for r in sorted(failed, key=lambda x: x.file):
            lines.append(f"\n{r.file}")
            for f in r.failures[:10]: lines.append(f"  • {f}")
            if len(r.failures) > 10: lines.append(f"  ... +{len(r.failures) - 10} more")
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Bulk validate PMC XML → Markdown")
    ap.add_argument("directory", type=Path, help="Folder containing .xml files")
    ap.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
    ap.add_argument("--report", type=Path, default=None, help="Write JSON report to file")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N files (0 = all)")
    args = ap.parse_args()

    if not args.directory.exists():
        print(f"Error: {args.directory} does not exist");
        sys.exit(1)

    xml_files = sorted(args.directory.rglob("*.xml"))
    if args.limit > 0: xml_files = xml_files[:args.limit]
    if not xml_files:
        print(f"No XML files found in {args.directory}");
        sys.exit(1)

    total = len(xml_files)
    print(f"Validating {total} files with {args.workers} workers...")
    print(f"{'─' * 60}")

    results: list = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(validate_single_file, str(p)): p for p in xml_files}
        done_count = 0
        for future in as_completed(futures):
            results.append(future.result())
            done_count += 1
            if done_count % 100 == 0 or done_count == total:
                elapsed = time.time() - start
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (total - done_count) / rate if rate > 0 else 0
                print(f"  [{done_count}/{total}] {rate:.1f} files/s ETA: {eta:.0f}s", end="\r")

    print(f"\n{'─' * 60}")
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    elapsed = time.time() - start
    stats = {
        "total": total, "passed": passed, "failed": failed,
        "pass_rate": round(100 * passed / total, 2) if total else 0,
        "elapsed_seconds": round(elapsed, 2),
        "files_per_second": round(total / elapsed, 1) if elapsed > 0 else 0,
    }

    print(f"\n  Total:    {total}\n  Passed:   {passed}\n  Failed:   {failed}")
    print(f"  Time:     {elapsed:.1f}s\n  Rate:     {total / elapsed:.1f} files/s")

    if failed > 0:
        print(f"\n{'─' * 60}\n  FAILURES ({failed}):\n{'─' * 60}")
        for r in sorted(results, key=lambda x: x.file):
            if not r.passed:
                print(f"\n  ✗ {r.file}")
                for f in r.failures[:5]: print(f"      • {f}")
                if len(r.failures) > 5: print(f"      ... +{len(r.failures) - 5} more")

    if args.report:
        json_path = args.report
        txt_path = args.report.with_suffix(".txt")
        write_json_report(results, json_path, stats)
        write_text_report(results, txt_path, stats)
        print(f"\n  JSON report: {json_path}\n  Text report: {txt_path}")
    else:
        txt_path = Path("validation_report.txt")
        write_text_report(results, txt_path, stats)
        print(f"\n  Report written to: {txt_path}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()