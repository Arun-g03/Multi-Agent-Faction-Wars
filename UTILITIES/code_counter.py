#!/usr/bin/env python3
"""
Code Line Counter Utility

This script counts the total lines of code in the project, excluding:
- The script itself
- Non-code files (images, logs, etc.)
- Generated files and directories
- Documentation files
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set

# File extensions to include in code count
CODE_EXTENSIONS = {
    '.py',    # Python
    '.js',    # JavaScript
    '.ts',    # TypeScript
    '.java',  # Java
    '.cpp',   # C++
    '.c',     # C
    '.h',     # C/C++ headers
    '.cs',    # C#
    '.php',   # PHP
    '.rb',    # Ruby
    '.go',    # Go
    '.rs',    # Rust
    '.swift', # Swift
    '.kt',    # Kotlin
    '.scala', # Scala
    '.sh',    # Shell scripts
    '.bat',   # Batch files
    '.ps1',   # PowerShell
    '.sql',   # SQL
    '.html',  # HTML
    '.css',   # CSS
    '.scss',  # SCSS
    '.sass',  # SASS
    '.vue',   # Vue.js
    '.jsx',   # React JSX
    '.tsx',   # TypeScript JSX
}

# Directories to exclude
EXCLUDE_DIRS = {
    '.git',
    '.vscode',
    '.vs',
    '__pycache__',
    'node_modules',
    'venv',
    'env',
    '.venv',
    '.env',
    'build',
    'dist',
    'target',
    'bin',
    'obj',
    'RUNTIME_LOGS',
    'Profiling_Stats',
    'saved_models',
    'IMAGES',
    'Grass Tiles',
    'aserpite',
    'gifs',
    'pngs',
}

# File patterns to exclude
EXCLUDE_FILES = {
    'main.spec',
    'redundant_functions.txt',
    'source.txt',
    'Source.txt',
    'Example.aseprite',
    'Example.png',
}

def is_code_file(file_path: Path) -> bool:
    """Check if a file should be counted as code."""
    return file_path.suffix.lower() in CODE_EXTENSIONS

def should_exclude_path(path: Path) -> bool:
    """Check if a path should be excluded from counting."""
    # Exclude directories
    if path.name in EXCLUDE_DIRS:
        return True
    
    # Exclude specific files
    if path.name in EXCLUDE_FILES:
        return True
    
    # Exclude hidden files and directories
    if path.name.startswith('.'):
        return True
    
    # Exclude this script itself
    if path.name == 'code_counter.py':
        return True
    
    return False

def count_lines_in_file(file_path: Path) -> int:
    """Count non-empty lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            # Count non-empty lines (excluding whitespace-only lines)
            return sum(1 for line in lines if line.strip())
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

def scan_project(project_root: Path) -> Dict[str, int]:
    """Scan the project and count lines of code by file type."""
    stats = {}
    total_lines = 0
    total_files = 0
    
    print(f"Scanning project: {project_root}")
    print("=" * 50)
    
    for file_path in project_root.rglob('*'):
        if file_path.is_file() and not should_exclude_path(file_path):
            if is_code_file(file_path):
                lines = count_lines_in_file(file_path)
                if lines > 0:
                    extension = file_path.suffix.lower()
                    stats[extension] = stats.get(extension, 0) + lines
                    total_lines += lines
                    total_files += 1
                    print(f"{file_path.relative_to(project_root)}: {lines} lines")
    
    return stats, total_lines, total_files

def print_summary(stats: Dict[str, int], total_lines: int, total_files: int):
    """Print a summary of the code count."""
    print("\n" + "=" * 50)
    print("CODE COUNT SUMMARY")
    print("=" * 50)
    
    # Sort by line count
    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
    
    for extension, lines in sorted_stats:
        percentage = (lines / total_lines) * 100 if total_lines > 0 else 0
        print(f"{extension:>8}: {lines:>6} lines ({percentage:>5.1f}%)")
    
    print("-" * 50)
    print(f"Total files: {total_files}")
    print(f"Total lines: {total_lines}")
    print("=" * 50)

def main():
    """Main function."""
    # Get the project root (parent of UTILITIES directory)
    script_path = Path(__file__)
    project_root = script_path.parent.parent
    
    print("Code Line Counter Utility")
    print("=" * 50)
    
    if not project_root.exists():
        print(f"Error: Project root not found at {project_root}")
        sys.exit(1)
    
    stats, total_lines, total_files = scan_project(project_root)
    print_summary(stats, total_lines, total_files)

if __name__ == "__main__":
    main() 