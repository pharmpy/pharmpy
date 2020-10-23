import difflib
import re

from markdown import markdown
import IPython


def markdown_html(string):
    return IPython.display.HTML(markdown(string))


def print_model_diff(model_ref, model_new):
    model_ref = str(model_ref).split('\n')
    model_new = str(model_new).split('\n')

    diffs = list(set(model_new) - set(model_ref))

    style_default = 'style=font-style:normal;font-size:87.5%;font-family:monospace;'

    style_insert = f'{style_default}background-color:#d6f3e8'
    style_replace = f'{style_default}background-color:#feffd2'

    str_full = ''

    for row in model_new:
        if row in diffs:
            close_matches = difflib.get_close_matches(row, model_ref, cutoff=0.75)
            if len(close_matches) > 0:
                str_full += f'<text {style_replace}>{row}</text><br>'
            else:
                str_full += f'<text {style_insert}>{row}</text><br>'
        else:
            str_full += f'<text {style_default}>{row}</text><br>'

    str_full = re.sub('\*', '\\\*', str_full)

    display(markdown_html(str_full))


def hover(hover_color="#ffff99"):
    return dict(selector="tr:hover",
                props=[("background-color", "%s" % hover_color)])


def print_df(df):
    styles = [
        hover(),
    ]
    return df.style.set_table_styles(styles)
