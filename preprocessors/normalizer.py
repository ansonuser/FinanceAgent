from collections import Counter
from rapidfuzz import fuzz

def build_frequency_map(all_keys):
    normalized = [normalize_html_key(k) for k in all_keys]
    freq = Counter(normalized)
    return freq

def build_canonical_map(freq, clusters):
    canonical_map = {}
    for group in clusters:
        canonical = max(group, key=lambda k: freq[k])  # 出現最多者
        for g in group:
            canonical_map[g] = canonical
    return canonical_map

def cluster_by_similarity(freq, threshold=90):
    keys = list(freq.keys())
    used = set()
    clusters = []
    for i, k1 in enumerate(keys):
        if i in used:
            continue
        group = [k1]
        for j, k2 in enumerate(keys):
            if j != i and fuzz.token_sort_ratio(k1, k2) >= threshold:
                group.append(k2)
                used.add(j)
        used.add(i)
        clusters.append(group)
    return clusters

def normalize_html_key(s):
    import re
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9\s&]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\b(revenues?|sales|net sales|total|segment|division|co|inc|ltd)\b', '', s)
    return s.strip()

def normalize_keys(all_keys, threshold=90):
    freq = build_frequency_map(all_keys)
    clusters = cluster_by_similarity(freq, threshold)
    canonical_map = build_canonical_map(freq, clusters)
    return canonical_map

if __name__ == "__main__":
    import os, json
    cur = "result"
    company = "MSFT"
    path = os.path.join(cur, company, "predictions")
    keys = []
    for f in os.listdir(path):
        # print(f)
        with open(os.path.join(path, f), "r", encoding="utf-8") as f:
            data = json.load(f)
            # print(data)
            keys += data["product_segments"].keys()
    print(normalize_keys(keys))