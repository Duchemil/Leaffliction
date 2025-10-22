import os
import sys
import argparse

#!/usr/bin/env python3
"""
clean.py

Remove all files in a directory tree that do NOT end with a given suffix.
Default suffix to keep: ').JPG'

Usage:
    python clean.py /path/to/dir --keep ").JPG" [--dry-run] [--force] [--verbose]
"""

def find_targets(root, keep_suffix):
        targets = []
        for dirpath, _, files in os.walk(root):
                for fn in files:
                        if not fn.endswith(keep_suffix):
                                targets.append(os.path.join(dirpath, fn))
        return targets

def parse_args():
        p = argparse.ArgumentParser(description="Remove files that don't end with a given suffix.")
        p.add_argument("root", nargs="?", default=".", help="Directory to clean (default: current dir)")
        p.add_argument("--keep", default=").JPG", help="Suffix to keep (default: ').JPG')")
        p.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting")
        p.add_argument("--force", "-f", action="store_true", help="Do not prompt for confirmation")
        p.add_argument("--verbose", "-v", action="store_true", help="Print each removed file")
        return p.parse_args()

def main():
        args = parse_args()
        root = os.path.abspath(args.root)
        if not os.path.isdir(root):
                print(f"Error: not a directory: {root}", file=sys.stderr)
                sys.exit(2)

        targets = find_targets(root, args.keep)
        if not targets:
                print("No files to remove.")
                return

        print(f"Found {len(targets)} file(s) not ending with '{args.keep}' under: {root}")
        if args.dry_run:
                for p in targets:
                        print("Would remove:", p)
                return

        if not args.force:
                ans = input("Proceed to delete these files? [y/N]: ").strip().lower()
                if ans not in ("y", "yes"):
                        print("Aborted.")
                        return

        removed = 0
        for path in targets:
                try:
                        os.remove(path)
                        removed += 1
                        if args.verbose:
                                print("Removed:", path)
                except Exception as e:
                        print(f"Failed to remove {path}: {e}", file=sys.stderr)

        print(f"Done. Removed {removed} file(s).")

if __name__ == "__main__":
        main()