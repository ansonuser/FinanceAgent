import os
import sys

def list_json_filenames(root, strip=''):
    """
    Recursively collect only JSON filenames (no path).
    Returns a set of filenames, e.g. {"config.json", "summary.json"}.
    """
    filenames = set()
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".json"):
                if strip:
                    f = f.replace(strip, '')
                filenames.add(f)
    return filenames


def compare_folders(folder1, folder2):
    files1 = list_json_filenames(folder1, '_prediction')
    files2 = list_json_filenames(folder2)

    only_in_1 = sorted(files1 - files2)
    only_in_2 = sorted(files2 - files1)
    common = sorted(files1 & files2)

    print(f"\n📁 Comparing folders (by filename only):")
    print(len(common))
    print(f"  Folder 1: {folder1}")
    print(f"  Folder 2: {folder2}\n")

    print("✅ Common JSON filenames:")
    for f in common or ["(none)"]:
        print("   ", f)

    print("\n❌ Only in Folder 1:")
    for f in only_in_1 or ["(none)"]:
        print("   ", f)

    print("\n❌ Only in Folder 2:")
    print(len(only_in_2))
    for f in only_in_2 or ["(none)"]:
        print("   ", f)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_json_folders.py <folder1> <folder2>")
        sys.exit(1)

    folder1, folder2 = sys.argv[1], sys.argv[2]

    if not os.path.isdir(folder1) or not os.path.isdir(folder2):
        print("Error: One or both paths are not valid directories.")
        sys.exit(1)

    compare_folders(folder1, folder2)
