from markdown import markdown
import IPython
import re

import difflib

Markdown = lambda string: IPython.display.HTML(markdown(string))


def print_model_diff(model_ref, model_new):
    model_ref = str(model_ref).split('\n')
    model_new = str(model_new).split('\n')

    diffs = list(set(model_new) - set(model_ref))

    style_default = 'style=font-style:normal;font-size:87.5%;font-family:monospace;'

    style_diff = f'{style_default}background-color:#feffd2'

    str_full = ''

    for row in model_new:
        if row in diffs:
            str_full += f'<text {style_diff}>{row}</text><br>'
        else:
            str_full += f'<text {style_default}>{row}</text><br>'

    str_full = re.sub('\*', '\\\*', str_full)

    display(Markdown(str_full))


def print_model_diff_colors(model_a, model_b):
    model_a = str(model_a)
    model_b = str(model_b)
    style = 'style=font-style:normal;font-size:87.5%;font-family:monospace;'

    style_insert = f'{style}background-color:#d6f3e8'
    style_replace = f'{style}background-color:#feffd2'
    style_delete = f'{style}background-color:#ffcfcc'

    str_full = ''
    matcher = difflib.SequenceMatcher(None, model_a, model_b)
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == 'equal':
            str_full += f'<text {style}>{model_a[a0:a1]}</text>'
        elif opcode == 'insert':
            str_full += f'<text {style_insert}>{model_b[b0:b1]}</text>'
        elif opcode == 'replace':
            str_full += f'<text {style_replace}>{model_b[b0:b1]}</text>'

    str_full = re.sub('\n', '<br>', str_full)
    str_full = re.sub('\*', '\\\*', str_full)

    display(Markdown(str_full))
