import os
import re

def remove_tsla_files(file_list):
    base_dir = r"D:\Side_projects\llm_cache_test\result\TSLA"

    for f in file_list:
        # Try to extract the 8-digit date (e.g., 20180930)
        match = re.search(r'(\d{8})', f)
        if not match:
            print(f"⚠️ Could not extract date from filename: {f}")
            continue

        date_str = match.group(1)
        year = int(date_str[:4]) + 1
        period_folder = f"{year}-01-01K"

        # Build both paths
        pred_path = os.path.join(base_dir, "predictions", f)
        cand_filename = f.replace("_prediction.json", "_candidated_chunks.json")
        cand_path = os.path.join(base_dir, "candidates", period_folder, "chunks", cand_filename)

        # Attempt deletion
        for path in [pred_path, cand_path]:
            if os.path.exists(path):
                os.remove(path)
                print(f"✅ Removed: {path}")
            else:
                print(f"⚠️ Not found: {path}")

if __name__ == "__main__":
    files =  []
    remove_tsla_files(files)

