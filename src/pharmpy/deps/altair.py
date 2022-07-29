import altair
from altair.utils.execeval import eval_block


def pharmpy_theme():
    return {
        'config': {
            'axis': {
                'labelFontSize': 11,
                'titleFontSize': 13,
            },
            'legend': {
                'labelFontSize': 12,
                'titleFontSize': 13,
            },
        }
    }


altair.themes.register('pharmpy', pharmpy_theme)
altair.themes.enable('pharmpy')
altair.data_transformers.disable_max_rows()

__all__ = ('altair', 'eval_block')
