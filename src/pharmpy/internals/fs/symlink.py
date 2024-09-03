import os
import subprocess
from pathlib import Path


def _create_junction(link_path: Path, target_path: Path):
    # DEVNULL needed to avoid WinError 6: The handle is invalid
    subprocess.check_call(
        f'mklink /J "{link_path}" "{target_path}" >nul 2>&1',
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def create_directory_symlink(link_path: Path, target_path: Path):
    # Will create a regular symlink on Linux and MacOS.
    # On Windows it will attempt a symlink and fallback to a junction
    # Note that junctions cannot have relative paths and might also fail on
    # network drives
    # Creating a symlink in Windows requires elevated privileges, but creating a junction
    # doesn't.
    if os.name == 'nt':
        try:
            link_path.symlink_to(target_path)
        except OSError:
            # symlink not supported
            _create_junction(link_path, target_path)
    else:
        link_path.symlink_to(target_path)
