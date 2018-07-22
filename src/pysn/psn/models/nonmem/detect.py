def detect(content):
    for line in content:
        if line.startswith('$PRO'):
            return True
    return False
