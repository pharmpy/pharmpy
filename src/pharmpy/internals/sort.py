import re


def sort_alphanum(strings):

    def keyfunc(s):
        tokens = [int(tok) if tok.isdigit() else tok for tok in re.findall(r"\d+|\D+", s)]
        return tokens

    return sorted(strings, key=keyfunc)
