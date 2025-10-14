import re
from bs4 import BeautifulSoup, Tag
from typing import List, Dict, Tuple
import json
from heapq import *
from bs4 import BeautifulSoup, Comment
import hashlib

cursor = [0]

def load_and_split_sec_html(path: str) -> List[Dict[str, str]]:
    """
    Load a 10-Q/10-K HTML file, extract sections (Item/Note), and flatten only tables.
    Returns list of {"anchor": <section header>, "text": <flattened section text>}.
    """

    # ========== Step 1️⃣ 讀取與解析 HTML ==========
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    
    if re.search(r"(?im)item\s+1\..{0,80}\|\s*\d{1,3}", html[:5000]):
        start = re.search(r"(?im)item\s+1\.?\s+(financial|consolidated)", html)
        if start:
            html = html[start.start():]
    soup = BeautifulSoup(html, "html5lib")

    # --- 移除不可見元素 ---
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

    for tag in soup(["font", "b", "i", "u", "a", "span"]):
        tag.unwrap()

    for table in soup.find_all("table"):
        if table.find_parent("table") is not None:
            table.unwrap()
    print("total tabls:", len(soup.find_all("table")))

    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    body = soup.body or soup  # fallback


    # ========== Step 2️⃣ 找出章節 anchor ==========
    anchors = []
    for tag in body.find_all(["p", "td", "div"], string=True):
        text = tag.get_text(" ", strip=True)
        
        # 匹配 Item/Note 開頭
        if re.match(r"(?i)^(item\s+\d+[a-z]?\b|note\s+\d+\b)", text):
            anchors.append(tag)

    if not anchors:  # 若未找到，可能是純文字
        print("⚠️ No anchors found; returning flattened text only.")
        return [{"anchor": "FULL_DOCUMENT", "text": flatten_only_tables(body, seen_tables)[0]}]

    # ========== Step 3️⃣ 以 anchor 為界切割章節 ==========
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

    # ========== Step 4️⃣ flatten table only ==========
    final_chunks = []
    table_seen = set()   # ✅ 新增：用來去重複，不重複處理相同表格

    for sec in sections:
        soup_sec = BeautifulSoup(sec["html"], "html5lib")  # ✅ 改：先 parse 這個 section 的 HTML
        tables = soup_sec.find_all("table")                # ✅ 改：找出 section 裡的每一個表格

        if not tables:
            continue

        for idx, tbl in enumerate(tables):  # ✅ 改：逐張表格處理，而不是整段一次
            # --- 去重：算 MD5 hash 來避免相同表被重複 flatten ---
            html_repr = re.sub(r"\s+", " ", str(tbl)).strip().lower()
            h = hashlib.md5(html_repr.encode()).hexdigest()  # ✅ 改：原本用 hash()，我改成 md5 更穩定
            if h in table_seen:
                continue
            table_seen.add(h)

            # --- 展開此表格 ---
            text, valid = flatten_only_tables(str(tbl), table_seen)  # ✅ 改：只傳入單一表格，而不是整段 section
            if not valid or len(text.strip()) < 50:                   # ✅ 改：略過過短或無效的表格
                continue

            # ✅ 改：每張表 append 一次，所以一個 section 可產生多筆結果
            final_chunks.append({
                "anchor": sec["anchor"],
                "text": text.strip()
            })

            print(f"✅ Added table from [{sec['anchor']}] (#{idx+1}) len={len(text)}")  # ✅ 可選 debug log

    print(f"✅ Total extracted tables: {len(final_chunks)}")
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
    last_table_close = pre_window.rfind("</table>", re.DOTALL | re.IGNORECASE)
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
        # if this tag remains unclosed, it likely wraps the table
        if stack:
            return left_bound + open_tags[0][1]

    # all balanced → discard pre-window
    return table_start


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
        return table_end + next_table_open

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



    haystack = str(soup)
    needle = str(table_node).lower()
    substring = haystack[-cursor[0]:].lower()  # your limited window
    start = substring.find(needle)
  
    shift = cursor[0]
    if start != -1:
        heappush(cursor, -start - cursor[0])
        if shift < 0:
            start = start - shift
    else:
        return ""
 
    end = start + len(str(table_node))
    left = find_balanced_left_boundary(str(soup), start, pre_chars)
    right = find_balanced_right_boundary(str(soup), end, post_chars)

    snippet = str(soup)[left:right]
    snippet = snippet.strip()

    if len(snippet) == 0:
        print("empty snippet")
        return ""

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

            if len(txt.split()) <= 50 and re.search(r"\(in\s+(millions|billions|thousands|percentages)", txt, re.I):
                parts.append(f"{txt}")
            continue

        # ---------- POST-CONTEXT (Footnotes) ----------
        if after_table and node.name != "table":
            # Common footnote patterns
            if re.search(r"\(in\s+(millions|billions|thousands|percentages)", txt, re.I) and len(txt.split()) <= 50:
                footnotes.append(f"{txt}")

            if max_footnotes == 0:
                break
            max_footnotes -= 1

    print("parts=", parts)
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
        print("pre =", pre)
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

def flatten_only_tables(html: str, seen_tables : set) -> Tuple[str, bool]:
    """
    Flatten each <table> exactly once and preserve surrounding text.
    Handles nested wrappers and duplicated references.
    """
    soup = BeautifulSoup(html, "html5lib")
    output_parts = []
    valid = False
    

    def traverse(node):
        nonlocal valid
        for elem in getattr(node, "children", []):
   
            # --- Table node ---
            if elem.name == "table":
                # Normalize representation to hash and deduplicate
                html_repr = re.sub(r"\s+", " ", str(elem)).strip()
                h = hashlib.md5(html_repr.encode()).hexdigest()
                if h in seen_tables:
                    continue
                seen_tables.add(h)             
                context_block = extract_table_with_text_context(soup, elem)
                output_parts.append(context_block)
                valid = True

                continue
            traverse(elem)
    traverse(soup)

    text = "\n\n".join(p for p in output_parts if p.strip())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, valid

def main(path : str, company_name : str, is_export : bool = True) -> List[dict[str, str]]:
    chunks = load_and_split_sec_html(path)

    print(f"Extracted {len(chunks)} sections.\n")
    # for c in chunks:
    #     print("🔹", c["anchor"])
        # print(c["text"][:100], "\n---\n")
    if is_export:
        with open(f"chunked_test_{company_name}.json", "w") as f:
            json.dump(chunks, f)
        print(f"Saving chunked_test_{company_name}.json to the disk.")
    return chunks


if __name__ == "__main__":
    import sys
    # assert len(sys.argv) == 3
    argv = sys.argv
    argv = [0, r'.\data\appl-d66145d10q.htm', r'aapl']
    path = argv[1] 
    company_name = argv[2]
    p = path 
    # r"D:\Side_projects\llm_cache_test\amzn-20240930.htm"
    chunks = main(path, company_name, is_export=True)
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
