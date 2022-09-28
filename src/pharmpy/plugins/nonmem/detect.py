import re


def detect_model(src):
    """Check if src represents a NONMEM control stream
    i.e. check if it is a file that contain $PRO
    """
    if not isinstance(src, str):
        return False
    is_control_stream = re.search(r'^\s*\$PRO', src, re.MULTILINE)
    return is_control_stream
