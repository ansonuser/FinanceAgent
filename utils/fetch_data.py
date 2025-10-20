import sys, re, requests, os
from schema.data import CompanyIndexEntry
import datetime
import time
from typing import Dict, Tuple
import tqdm
import tempfile
from arelle import CntlrCmdLine


PREFIX_URL = "https://www.sec.gov/Archives/"
HEADERS = {"User-Agent": "MyApp contact@example.com"}
TIME_TRACKS = []
TICKER_PATH = "\\configs\\company_tickers.json"

def clean_inline_xbrl_html(body: str) -> str:
    """
    Clean inline XBRL HTML using Arelle's Viewer plugin.
    If not an iXBRL file, returns the original text unchanged.
    Works in memory via a temporary file.
    """
    if not re.search(r"<ix:", body, re.I):
        return body

    tmp = None
    try:
        # --- Save to temporary file because Arelle requires a real URI
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".htm", mode="w", encoding="utf-8")
        tmp.write(body)
        tmp.flush()
        tmp.close()

        out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        out_tmp.close()
        # pluginClassLoader.loadModule("xbrl.view.xhtml.ViewerCmd")
        # --- Run Arelle Viewer plugin
        # cntlr = CntlrCmdLine.CntlrCmdLine()
        sys_argv_backup = sys.argv
        sys.argv = [
            "arelleCmdLine",
            "--file", tmp.name,
            "--plugins", "EDGAR/render",
            "--viewFile", out_tmp.name,
        ]
        CntlrCmdLine.main()  # t
        # --- Read back cleaned HTML
        with open(out_tmp.name, encoding="utf-8", errors="ignore") as f:
            html_clean = f.read()

        # --- Cleanup temp files
        os.remove(tmp.name)
        os.remove(out_tmp.name)

        return html_clean or body

    except Exception as e:
        print("⚠️ Inline XBRL cleaning failed:", e)
        if tmp and os.path.exists(tmp.name):
            os.remove(tmp.name)
        return body



def extract_xbrl_from_txt(file=None, filename=None, outdir=""):
    if file is None:
        assert filename is not None
        with open(filename, encoding="utf-8") as f:
            text = f.read()
    else:
        text = file
   
    # Find each <DOCUMENT>...</DOCUMENT> block
    docs = re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", text, flags=re.S | re.I)

    for d in docs:
        # Extract <TYPE>, <FILENAME>, <TEXT>
        m_type = re.search(r"<TYPE>(.*?)\n", d)
        m_name = re.search(r"<FILENAME>(.*?)\n", d)
        m_text = re.search(r"<TEXT>(.*?)</TEXT>", d, flags=re.S | re.I)
        # print(m_name)
        if not (m_type and m_name and m_text):
            continue

        ftype = m_type.group(1).strip().upper()
        fname = m_name.group(1).strip()
        body  = m_text.group(1).lstrip()
        # print(ftype)
        # Keep only XBRL-related files
        if ftype.startswith("EX-101") or re.match("10-Q", ftype) or re.match("10-K", ftype):
            if body.startswith("<XBRL>"):
                body = re.sub(r"^<XBRL>\s*", "", body)
                body = re.sub(r"</XBRL>\s*$", "", body)
            body = clean_inline_xbrl_html(body)
            path = os.path.join(outdir, fname)
            os.makedirs(outdir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(body)
            print("✅ saved", fname)

def load_company_tickers(tickers_path=None) -> list[CompanyIndexEntry]:
    if tickers_path is None:
        tickers_path  = TICKER_PATH

    with open(tickers_path, "r", encoding="utf-8") as f:
        import json
        tickers = json.load(f)

    from typing import List
    entries: List[CompanyIndexEntry] = []
    for _, entry in tickers.items():
        cik = f"{int(entry['cik_str']):010d}"
        ticker = (entry.get("ticker") or "").upper()
        name = entry.get("title") or ""
        entries.append(CompanyIndexEntry(cik=cik, ticker=ticker, name=name))
    return entries


def load_master(master_dir="master_files")->Dict[str, str]:
    """Load master index file for a given year and quarter.
    """
    res = {}
    for year in range(STARTY, ENDY):
        for quarter in range(1, 5):
            filename = f"master_idx_{year}_{quarter:02d}.txt"
            path = os.path.join(master_dir, filename)
            with open(path, encoding="utf-8") as f:
                lines = f.read()
            res[filename] = lines
    return res

def search_cik_in_master(cik: str, year: int, quarter: int)->Tuple[bool, str]:
    """Search for a given CIK in the master index file for a given year and quarter.
    Return the line if found, else return empty string.
    """
    filename = f"master_idx_{year}_{quarter:02d}.txt"
    lines = TABLE.get(filename, "")
    cik = cik.lstrip('0')
    for line in lines.splitlines():
        if line.startswith(cik):
            sp = line.split("|")
            if len(sp) < 4:
                continue
            if sp[2] == "10-Q": 
                K = False
            elif sp[2] == "10-K":
                K = True
            else:
                continue
            return K, line.split('|')[-1].strip()
    return False, ""

def download(url:str, save_path:str)->Tuple[bool, str]:
    try:
        res = requests.get(url, headers=HEADERS).text
    except:
        print(f"Request Error at:{url}")
        return False, ""
    
    with open(save_path, 'w') as f:
        f.write(res)
    return True, res

def main():
    company_index = load_company_tickers()[:NUM]
    print(company_index)
    for c_idx in tqdm.tqdm(company_index):
        if c_idx.ticker == 'TSLA':
            continue
        cik = c_idx.cik
        stock_name = c_idx.ticker
        cur_folder = './data_new/' + stock_name
        if not os.path.exists(cur_folder):
            os.makedirs(cur_folder)
        
        for year in range(STARTY, ENDY):
            for quarter in range(1, 5):
                cur_path = cur_folder + '/' + f"{year}-{quarter:02d}-01"
                K, url = search_cik_in_master(cik, year, quarter)
                 
                if url:
                    if len(TIME_TRACKS) >= 8:
                        time_diff = datetime.datetime.now() - TIME_TRACKS[0]
                        if time_diff.total_seconds() <= 2:
                            sleep_time = 2 - time_diff.total_seconds()
                            time.sleep(sleep_time)
                        TIME_TRACKS.pop(0)
                    cur_path += 'K' * K 
                    if not os.path.exists(cur_path):
                        os.makedirs(cur_path)
                    r, data = download(PREFIX_URL + '/' + url, save_path=cur_path + '/' + url.split('/')[-1])
                    TIME_TRACKS.append(datetime.datetime.now())
                    if r:
                        print(f"Download {stock_name}:{year}-{quarter:02d} successfully!")
                        extract_xbrl_from_txt(data, outdir=cur_path)
                    else:
                        print(f"Fail to download {stock_name}:{year}-{quarter:02d}")

if __name__ == "__main__":
    STARTY, ENDY = 2025, 2026
    NUM = 5
    TABLE = load_master()

    main()
    # path = r"data\GOOGL"
    # for cur_d in os.listdir(path):
    #     cur_f = os.path.join(path, cur_d)
    #     for f in os.listdir(cur_f):
    #         if f.endswith("txt"):
    #             p = os.path.join(cur_f, f)
    #             extract_xbrl_from_txt(filename=p, outdir=cur_f)
            

    

