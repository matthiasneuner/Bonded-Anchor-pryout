#!/usr/bin/env python3
"""
Convert Abaqus concrete mesh into dual-mesh format required for UEL analysis.

This script duplicates the element section of an Abaqus input file:
1. Creates visualization-only elements (C3D20R) with ELSET=dummy and original element IDs.
2. Creates User Elements (U004) with ELSET=concrete and element IDs offset by offset (default: 900000).
3. Preserves all nodes, node sets, element sets, and surface definitions verbatim.

Usage:
    python create_dual_mesh.py [--input concrete.inp] [--offset 900000]
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert single-element-block Abaqus mesh into dual-mesh format for UEL analysis."
    )
    parser.add_argument(
        "-i", "--input",
        default="concrete.inp",
        type=str,
        help="Input Abaqus mesh file (default: concrete.inp)"
    )
    parser.add_argument(
        "-o", "--offset",
        default=900000,
        type=int,
        help="Element ID offset for U004 elements (default: 900000)"
    )
    return parser.parse_args()


def process_element_block(element_lines, fout, offset):
    """
    Process collected element data lines and write out:
    1. C3D20R dummy block with original element IDs.
    2. U004 concrete block with element IDs offset by offset.
    
    Handles both single-line and multi-line (continuation line) element formats.
    Returns the total number of elements processed.
    """
    # 1. Write dummy element header and original connectivity lines verbatim
    fout.write("*ELEMENT, TYPE=C3D20R, ELSET=dummy\n")
    for line in element_lines:
        fout.write(line)

    # 2. Write U004 element header and offset connectivity lines
    fout.write("*ELEMENT, TYPE=U004, ELSET=concrete\n")
    
    elem_count = 0
    expecting_continuation = False
    nodes_needed = 0

    for line in element_lines:
        stripped = line.strip()
        if not stripped:
            fout.write(line)
            continue

        if not expecting_continuation:
            elem_count += 1
            parts = line.split(",", 1)
            elem_id_str = parts[0]
            elem_id = int(elem_id_str.strip())
            new_elem_id = elem_id + offset
            width = len(elem_id_str)
            new_elem_id_str = f"{new_elem_id:>{width}}"
            
            modified_line = new_elem_id_str + "," + parts[1]
            fout.write(modified_line)

            # Count node entries on this line
            node_entries = [p for p in parts[1].strip().rstrip(",").split(",") if p.strip()]
            num_nodes = len(node_entries)
            if num_nodes < 20:
                expecting_continuation = True
                nodes_needed = 20 - num_nodes
            else:
                expecting_continuation = False
                nodes_needed = 0
        else:
            # Continuation line (node IDs only) written verbatim
            fout.write(line)
            node_entries = [p for p in line.strip().rstrip(",").split(",") if p.strip()]
            num_nodes = len(node_entries)
            nodes_needed -= num_nodes
            if nodes_needed <= 0:
                expecting_continuation = False
                nodes_needed = 0

    return elem_count


def convert_dual_mesh(input_path: str, output_path: str, offset: int):
    """
    Reads input mesh file, converts *ELEMENT block into dual-mesh format,
    and writes to output_path.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading input file: {input_path}")
    print(f"Using element ID offset: {offset}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    element_lines = []
    in_element_section = False
    total_elements = 0

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            if line.startswith("*"):
                # Check if keyword line starts element section
                if line.strip().upper().startswith("*ELEMENT"):
                    in_element_section = True
                    element_lines = []
                    continue
                elif in_element_section:
                    # End of element section reached (keyword or comment line starting with '*')
                    total_elements += process_element_block(element_lines, fout, offset)
                    in_element_section = False
                    element_lines = []
                    fout.write(line)
                    continue

            if in_element_section:
                element_lines.append(line)
            else:
                fout.write(line)

        # Handle case where element block extends to end of file
        if in_element_section and element_lines:
            total_elements += process_element_block(element_lines, fout, offset)

    print(f"Processed {total_elements} elements.")
    print(f"Successfully generated dual mesh output: {output_path}")


def main():
    args = parse_args()

    # Determine script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Resolve input path relative to script directory if relative
    if os.path.isabs(args.input):
        input_path = args.input
    else:
        input_path = os.path.normpath(os.path.join(script_dir, args.input))

    # Output path inside 'modified/' subdirectory relative to input file location
    output_dir = os.path.join(os.path.dirname(input_path), "modified")
    output_path = os.path.join(output_dir, os.path.basename(input_path))

    convert_dual_mesh(input_path, output_path, args.offset)


if __name__ == "__main__":
    main()
