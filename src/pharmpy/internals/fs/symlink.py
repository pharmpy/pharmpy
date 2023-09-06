import os
import subprocess
from pathlib import Path


def create_directory_symlink(link_path: Path, target_path: Path):
    # Will create a regular symlink on Linux and MacOS and a junction on Windows
    # Creating a symlink in Windows requires elevated privileges, but creating a junction
    # doesn't. The junction is mostly equivalent to symlinks so should be fine.
    if os.name == 'nt':
        subprocess.check_call(f'mklink /J "{link_path}" "{target_path}" >nul 2>&1', shell=True)
    else:
        link_path.symlink_to(target_path)
