"""
This file is used to chunk the document by table with outwardly expanding. 
"""

import json, re
from typing import List, Dict
from bs4 import BeautifulSoup, Tag, Comment
from heapq import *
import hashlib
import os
COUNT = 0
cursor = [0]

def load_and_split_sec_html(path: str) -> List[Dict[str, str]]:
    """
    Load a 10-Q/10-K HTML file, extract sections (Item/Note), and flatten only tables.
    Returns list of {"anchor": <section header>, "text": <flattened section text>}.
    """

    # ========== Read file and remove table of content ==========
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    
    if re.search(r"(?im)item\s+1\..{0,80}\|\s*\d{1,3}", html[:5000]):
        start = re.search(r"(?im)item\s+1\.?\s+(financial|consolidated)", html)
        if start:
            html = html[start.start():]
    soup = BeautifulSoup(html, "html5lib")

    # Remove useless information in html
    for tag in soup(["script", "style", "meta", "link", "head"]):
        tag.decompose()
    
    visual_attrs = {
        "style", "align", "valign", "width", "height",
        "cellpadding", "cellspacing", "border",
        "bgcolor", "color", "face", "size",
        "nowrap", "lang", "xml:lang", "class", "title"
    }

    for tag in soup.find_all(True):
        for attr in list(tag.attrs.keys()):
            if attr in visual_attrs:
                tag.attrs.pop(attr, None)

    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # flatten layer
    for tag in soup(["font", "b", "i", "u", "a", "span"]):
        tag.unwrap()

    for table in soup.find_all("table"):
        if table.find_parent("table") is not None:
            table.unwrap()

    print("total tabls:", len(soup.find_all("table")))


    body = soup.body or soup  # fallback
    seen_tables = set()   # prevent re-flattening identical tables

    # ========== Identify anchors ==========
    anchors = []
    for tag in body.find_all(["p", "td", "div"], string=True):
        text = tag.get_text(" ", strip=True)
        
        # Match headers like Item/Note
        if re.match(r"(?i)^(item\s+\d+[a-z]?\b|note\s+\d+\b)", text):
            anchors.append(tag)

    if not anchors:  
        print("No anchors found; returning flattened text only.")
        return [{"anchor": "FULL_DOCUMENT", "text":
                  flatten_only_tables({"anchor" : "NO ANCHOR", "html" : str(body)}, seen_tables)[0]}]

    # ========== Extact information by anchors ==========
    sections = []
    for i, anchor in enumerate(anchors):
        start = anchor
        end = anchors[i + 1] if i + 1 < len(anchors) else None

        elems = []
        for elem in start.next_elements:
            if end and elem == end:
                break
            if getattr(elem, "name", None) in ["p", "div", "table"]:
                elems.append(str(elem))

        html_block = "\n".join(elems)
        sections.append({
            "anchor": anchor.get_text(" ", strip=True),
            "html": html_block
        })

    # ========== Convert html table to markdown table ==========
    final_chunks = []
    for sec in sections:
        final_chunks += flatten_only_tables(sec, seen_tables)
    print("len of seens tables", len(seen_tables))
    return final_chunks

def find_balanced_left_boundary(full_html: str,
                                table_start: int,
                                pre_chars: int = 800) -> int:
    """
    Find the earliest valid opening tag before the table that:
      - is a complete tag (not partial),
      - is NOT inside another <table> ... </table>,
      - and is within pre_chars.
    If none found, return table_start (discard pre-window).
    """

    left_bound = max(0, table_start - pre_chars)
    pre_window = full_html[left_bound:table_start].lower()

    # If another table appears in the pre-window, stop at its closing boundary
    last_table_close = pre_window.rfind("</table>")
    if last_table_close != -1:
        # We found a prior table; discard content before its end
        return left_bound + last_table_close + len("</table>")

    # --- collect open/close tags ---
    openings = list(re.finditer(r"<([a-zA-Z0-9]+)(\s[^>]*)?>", pre_window))
    closings = list(re.finditer(r"</([a-zA-Z0-9]+)>", pre_window))

    if not openings:
        return table_start  # no open tags at all

    open_tags = [(m.group(1).lower(), m.start()) for m in openings]
    close_tags = [m.group(1).lower() for m in closings]

    # --- balance matching tags to find first unclosed open tag ---
    stack = []
    for tag, pos in open_tags:
        stack.append(tag)
        while close_tags and stack and close_tags[0] == stack[-1]:
            stack.pop()
            close_tags.pop(0)
        # complete sections
        if stack:
            return left_bound + open_tags[0][1]
    if len(open_tags) >= 2:
        return left_bound + open_tags[-2][1]
    else:
        return table_start - 100  # fall back


def find_balanced_right_boundary(full_html: str,
                                 table_end: int,
                                 post_chars: int = 1500) -> int:
    """
    Find a safe right boundary after the table.
    Stops at the first closing tag that completes the current section,
    but does NOT cross into the next <table>.
    """
    n = len(full_html)
    right_bound = min(n, table_end + post_chars)
    post_window = full_html[table_end:right_bound].lower()

    # stop if another table starts
    next_table_open = post_window.find("<table")
    if next_table_open != -1:
        return table_end + next_table_open - 1

    max_pos = 0
    # otherwise find the first closing container after the table
    for tagname in ["div", "section", "p"]:
        matches = list(re.finditer(fr"</{tagname}>", post_window, re.I))
        if matches:
            m = matches[-1]   # 
            max_pos = max(table_end + m.end(), max_pos)
     
    # fallback: the window end
    return max(right_bound, max_pos)


def extract_table_with_text_context(soup: BeautifulSoup | Tag,
                                    table_node: Tag,
                                    pre_chars: int = 1500,
                                    post_chars: int = 1500,
                                    max_footnotes: int = 5) -> str:
    """
    Extract <table> along with nearby explanatory text before and footnotes after it.
    - Captures pre-context (e.g., '(in millions)')
    - Captures post-context footnotes ((1), (2), 'includes', etc.)
    """
    assert table_node.name == "table"
    
    haystack = re.sub(r"\s+", " ", str(soup).lower().replace("\xa0", " "))
    needle = re.sub(r"\s+", " ", str(table_node).lower().replace("\xa0", " "))
    local_start = haystack.find(needle, cursor[0]) # your limited window
    
    if local_start == -1:
        local_start = haystack.find(needle)
        if local_start == -1:
            return ""
        
    # expand table size to window size (left, right) 
    global_start = local_start
    global_end   = global_start + len(needle)
    left = find_balanced_left_boundary(haystack , global_start, pre_chars)
    right = find_balanced_right_boundary(haystack , global_end, post_chars)
    cursor[0] = max(cursor[0], global_end)
    snippet = haystack[left:right]
    snippet = snippet.strip()

    count = len(re.findall(r"<\s*table\b", snippet, re.IGNORECASE))
    # normal should contain exact one table
    if count != 1:
        print(snippet)
        print("count = ", count)
        assert count == 1
    if len(snippet) == 0:
        print("empty snippet")
        return ""
  
    # --- Parse local snippet ---
    soup2 = BeautifulSoup(snippet, "html5lib")

    parts = []
    found_table = False
    footnotes = []
    after_table = False

    for node in soup2.find_all(["p", "div", "span", "table"]):
        txt = node.get_text(" ", strip=True)
        if not txt:
            continue
    
        # ---------- PRE-CONTEXT ----------
        if not found_table:
            if node.name == "table":
                parts.append(str(node))
                found_table = True
                after_table = True
                continue

            if len(txt.split()) <= 50 and re.search(r"in\s+(million|billion|thousand|percentage)s?", txt, re.I):
                parts.append(f"{txt}")
            continue

        # ---------- POST-CONTEXT (Footnotes) ----------
        if after_table and node.name != "table":
            # Common footnote patterns
            if re.search(r"in\s+(million|billion|thousand|percentage)s?", txt, re.I) and len(txt.split()) <= 50:
                footnotes.append(f"{txt}")

            if max_footnotes == 0:
                break
            max_footnotes -= 1
    if not found_table:
        print("Empty table found")
        return ""
        # assert found_table
    
    # Assemble: pre + table + footnotes
    pre, pos = [], []
    res = ""
    seen = set()
    for p in parts:
        if "<table" in p.lower():
            break
        if p not in seen:
            seen.add(p)
            pre.append(p)
    if pre:
        pre = "\n".join(pre) + "\n"
    else:
        pre = ""
   
    res += pre + flatten_table(table_node)
    seen.clear()
    if footnotes:
        for f in footnotes:
            if f not in seen:
                seen.add(f)
                pos.append(f)

        res += "\n" + "\n".join(footnotes)

    return '<table>\n'+ res + '\n</table>'


def flatten_table(table):
    """
    html format to markdown
    """
    lines = []
    for tr in table.find_all("tr"):
        cells = [
            td.get_text(" ", strip=True)
            for td in tr.find_all(["td", "th"])
            if td.get_text(strip=True)
        ]
        if cells:
            line = " | ".join(cells)
            line = re.sub(r"\s*\|\s*", " | ", line.strip())
            lines.append(line)
    clean_lines = [
        l for l in lines if re.match(r"[A-Za-z0-9,.\(\)]", l)
    ]
    return "\n".join(clean_lines) 

def flatten_only_tables(sec: List[Dict[str, str]], seen_tables: set)->List[dict[str, str]]:
    """
    extact table for each chunk
    """
    global COUNT
    chunks = []
    soup_sec = BeautifulSoup(sec["html"], "html5lib")
    tables = soup_sec.find_all("table")
    COUNT += len(tables)

    for tbl in tables:
    
        # remove duplication
        h = hashlib.md5(str(tbl).encode("utf-8")).hexdigest()
        if h in seen_tables:
            continue
        seen_tables.add(h)

        block = extract_table_with_text_context(soup_sec, tbl)
        text = block.strip()
        if len(text) > 200:  # garbage chunk if size is too small
            chunks.append({
                "anchor": sec["anchor"],
                "text": text
            })
    return chunks


def run_single(path : str, is_export : bool = True, dest : str = "preprocessed") -> List[dict[str, str]]:
    splits = path.split(os.sep)
    company_name = splits[1]
    date_str = splits[2]
    filename = splits[3]
    full_path = os.path.join(dest, company_name, date_str, filename.replace("htm", "json"))
    if os.path.exists(full_path):
        print(f"{full_path} existed!")
        return []
    chunks = load_and_split_sec_html(path)

    print(f"Extracted {len(chunks)} sections.\n")
    for c in chunks:
        print("🔹", c["anchor"])
    

    if is_export:
        if not os.path.exists(os.path.join(dest, company_name, date_str)):
            os.makedirs(os.path.join(dest, company_name, date_str))
        
        with open(full_path, "w") as f:
            json.dump(chunks, f)
        print(rf"Saving {full_path} to the disk.")
    return chunks

def main(path: str, is_export: bool = True):

    for parent,_,files in os.walk(path):
        for file in files:
            if file.endswith("htm"):
                # print(f"Processing on {file}")
                file_path = os.path.join(parent, file)
                run_single(file_path, is_export)





if __name__ == "__main__":
    import sys
    single_mode = False
    company_name = None
    argv = sys.argv
    # argv = [0, r'.\data\appl-d66145d10q.htm', r'aapl']
    path = argv[1] # src folder
    if len(argv) == 3:
        company_name = argv[2]

    if single_mode:
        assert company_name is not None
        p = path
        # r"D:\Side_projects\llm_cache_test\amzn-20240930.htm"
        chunks = run_single(path, company_name, is_export=True)
        raw = ""
        for d in chunks:
            raw += d["text"]
        matches = re.finditer(r"<table.*?>.*?</table>", raw, re.DOTALL | re.IGNORECASE)
        size = []
        count = 1
    
        for m in matches:
            if m.end() - m.start() > 150:
                print(f"*******[ID={count}]*********")
                print(raw[m.start() : m.end()])
                count += 1
    else:
        main(path)
