from markdown import markdown
import IPython


def markdown_html(string):
    return IPython.display.HTML(markdown(string))


def print_model_diff(model_ref, model_new):
    model_ref = str(model_ref).split('\n')
    model_new = str(model_new).split('\n')

    diffs = list(set(model_new) - set(model_ref))

    css_default = 'style=font-style:normal;font-size:80%;font-family:monospace;'

    # Hex codes:
    #    * Green:  #D6F3E8
    #    * Blue:   #DEF4FD
    #    * Yellow: #FEFFD2
    css_diff = f'{css_default}background-color:#DEF4FD'

    str_full = '<p style=line-height:120%>'

    for row in model_new:
        if row in diffs:
            str_full += f'<text {css_diff}>{row}</text><br>'
        else:
            str_full += f'<text {css_default}>{row}</text><br>'

    str_full += '</p>'

    display(markdown_html(str_full))
