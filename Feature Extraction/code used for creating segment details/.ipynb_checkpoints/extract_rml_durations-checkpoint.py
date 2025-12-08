#!/usr/bin/env python3
"""
extract_rml_durations.py

Example usage: python extract_rml_durations.py /path/to/rml_directory -o durations.csv

Scan a directory (recursively) for .rml files, parse each, extract the <Duration>
that is a direct child of each <Session>, and write a CSV with:
    filename (without .rml) , duration

If a file has multiple <Session> elements, the script writes one row per Session,
and appends _1, _2, ... to the filename stem for uniqueness.
"""

import argparse
import csv
from pathlib import Path
import xml.etree.ElementTree as ET
import sys

def local_name(tag: str) -> str:
    """Return the local (non-namespace) name of an XML tag."""
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag

def find_direct_child_by_localname(parent: ET.Element, name: str):
    """Return the first direct child of `parent` whose local tag name matches `name`."""
    for child in list(parent):
        if local_name(child.tag) == name:
            return child
    return None

def extract_session_durations_from_file(path: Path):
    """
    Parse XML file and return a list of durations (strings) for each <Session> found.
    Only the <Duration> that is a direct child of <Session> is returned.
    """
    durations = []
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"ERROR: failed to parse {path}: {e}", file=sys.stderr)
        return durations

    # Find all Session elements regardless of namespace
    sessions = [elem for elem in root.iter() if local_name(elem.tag) == 'Session']

    for session in sessions:
        dur_elem = find_direct_child_by_localname(session, 'Duration')
        if dur_elem is not None and dur_elem.text is not None:
            durations.append(dur_elem.text.strip())
        else:
            # No direct Duration child found; append empty string to keep alignment
            durations.append('')
    return durations

def main():
    parser = argparse.ArgumentParser(description="Extract <Duration> (direct child of <Session>) from .rml files and write CSV.")
    parser.add_argument('input_dir', type=str, help='Directory containing .rml files (will search recursively)')
    parser.add_argument('-o', '--output', type=str, default='durations.csv', help='Output CSV file path')
    parser.add_argument('--utf8', action='store_true', help='Write CSV as UTF-8 (default platform encoding otherwise)')
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"ERROR: input_dir {input_path} does not exist or is not a directory", file=sys.stderr)
        sys.exit(2)

    rml_files = sorted(input_path.glob('**/*.rml'))
    if not rml_files:
        print("No .rml files found.", file=sys.stderr)
        sys.exit(0)

    out_path = Path(args.output)
    write_encoding = 'utf-8' if args.utf8 else None

    with out_path.open('w', newline='', encoding=write_encoding) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'duration'])
        for rml in rml_files:
            durations = extract_session_durations_from_file(rml)
            if not durations:
                # write a row with empty duration so file is represented
                writer.writerow([rml.stem, ''])
                continue

            if len(durations) == 1:
                writer.writerow([rml.stem, durations[0]])
            else:
                # multiple sessions: write one row per session, append index to filename
                for i, d in enumerate(durations, start=1):
                    writer.writerow([f"{rml.stem}_{i}", d])

    print(f"Wrote durations for {len(rml_files)} files to {out_path}")

if __name__ == '__main__':
    main()
