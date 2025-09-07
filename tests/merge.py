# merge_py_to_txt.py
import glob
import os
import sys
from datetime import datetime

def main():
    # Output file name: pass as first arg, else default
    out_path = sys.argv[1] if len(sys.argv) > 1 else "merged_python_files.txt"

    # Gather all .py files in the current directory (sorted for determinism)
    py_files = sorted(f for f in glob.glob("*.py") if os.path.isfile(f))

    if not py_files:
        print("No .py files found in the current directory.")
        return

    # If you want to exclude this script itself, uncomment the next line:
    # py_files = [f for f in py_files if f != os.path.basename(__file__)]

    with open(out_path, "w", encoding="utf-8") as out:
        out.write("# Merged Python files\n")
        out.write(f"# Created: {datetime.now().isoformat(timespec='seconds')}\n")
        out.write(f"# Directory: {os.path.abspath(os.getcwd())}\n")
        out.write("# Files included:\n")
        for f in py_files:
            out.write(f"#   - {f}\n")
        out.write("\n")

        for f in py_files:
            out.write(f"# ===== BEGIN: {f} =====\n\n")
            with open(f, "r", encoding="utf-8", errors="replace") as src:
                for line in src:
                    out.write(line)
            # Ensure a trailing newline between files
            out.write(f"\n# ===== END: {f} =====\n\n")

    print(f"Merged {len(py_files)} files into: {out_path}")

if __name__ == "__main__":
    main()
