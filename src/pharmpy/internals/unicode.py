import math

mathematical_script_capital_n = 'N'  # 𝒩  would be better, but due to bug in
# Qt rendering (i.e. spyder, rstudio) gets misaligned


class Grid:
    def __init__(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols
        self._grid = [[None for x in range(ncols)] for y in range(nrows)]

    def set(self, row, col, value):
        self._grid[row][col] = value

    def _column_widths(self):
        """Calculate the number of characters needed for each column"""
        widths = []
        for col in range(self.ncols):
            maxlen = 0
            for row in range(self.nrows):
                e = self._grid[row][col]
                if e:
                    maxlen = max(maxlen, e.minimum_width)
            widths.append(maxlen)
        return widths

    def _row_height(self, row):
        maxheight = 0
        for col in range(self.ncols):
            e = self._grid[row][col]
            if e:
                maxheight = max(maxheight, e.minimum_height)
        return maxheight

    @staticmethod
    def _pad(lines, height):
        n = len(lines)
        if n == height:
            return lines
        else:
            length = len(max(lines, key=len))
            add = height - n
            before = add // 2
            after = add // 2
            if before + after != add:
                before += 1
            s = ' ' * length
            padded = [s] * before + lines + [s] * after
            return padded

    def __str__(self):
        output = ''
        colwidths = self._column_widths()
        for row in range(self.nrows):
            height = self._row_height(row)
            for line in range(height):
                for col in range(self.ncols):
                    width = colwidths[col]
                    e = self._grid[row][col]
                    if e:
                        lines = e.lines(width)
                        padded = Grid._pad(lines, height)
                        output += padded[line]
                    else:
                        output += ' ' * width
                output += '\n'
        return output


def _pad_on_sides(n, char=' '):
    if n <= 0:
        return '', ''
    n_before = n // 2
    n_after = n // 2
    if n_before + n_after != n:
        n_after += 1
    return char * n_before, char * n_after


class Box:
    def __init__(self, text):
        self.text = text.split('\n')

    @property
    def minimum_width(self):
        max_row_len = max(len(s) for s in self.text)
        return max_row_len + 2

    @property
    def minimum_height(self):
        return len(self.text) + 2

    def lines(self, width):
        lines = []
        upper = '┌' + '─' * (width - 2) + '┐'
        lines.append(upper)

        for row in self.text:
            before, after = _pad_on_sides(width - len(row) - 2)
            mid = '│' + before + row + after + '│'
            lines.append(mid)

        lower = '└' + '─' * (width - 2) + '┘'
        lines.append(lower)
        return lines


class Arrow:
    def __init__(self, text, right=True):
        self.text = text
        self.right = right

    @property
    def minimum_width(self):
        return len(self.text) + 3

    @property
    def minimum_height(self):
        return 1

    def lines(self, width):
        before, after = _pad_on_sides(width - self.minimum_width, char='─')
        if self.right:
            return ['─' * 2 + before + self.text + after + '→']
        else:
            return ['←' + before + self.text + after + '─' * 2]


class VerticalArrow:
    def __init__(self, text, down=True):
        self.text = text
        self.down = down

    @property
    def minimum_width(self):
        return len(self.text)

    @property
    def minimum_height(self):
        return 3

    def lines(self, width):
        extra_before, extra_after = _pad_on_sides(width - self.minimum_width)
        n = len(self.text) / 2
        before = extra_before + ' ' * math.floor(n)
        after = extra_after + ' ' * math.ceil(n)
        if self.down:
            return [
                before + '│' + after,
                extra_before + self.text + extra_after,
                before + '↓' + after,
            ]
        else:
            return [
                before + '↑' + after,
                extra_before + self.text + extra_after,
                before + '│' + after,
            ]


class DualVerticalArrows:
    def __init__(self, up_text, down_text):
        self.up_text = up_text
        self.down_text = down_text

    @property
    def minimum_width(self):
        return len(self.up_text) + len(self.down_text) + 1

    @property
    def minimum_height(self):
        return 3

    def lines(self, width):
        nbefore = len(self.up_text) // 2
        nafter = len(self.down_text) // 2
        nmiddle = width - 2 - nbefore - nafter
        before = ' ' * nbefore
        after = ' ' * nafter
        middle = ' ' * nmiddle
        first_line = before + '↑' + middle + '│' + after
        last_line = before + '│' + middle + '↓' + after
        middle_spaces = ' ' * (width - self.minimum_width + 1)
        middle_line = self.up_text + middle_spaces + self.down_text
        return [first_line, middle_line, last_line]


def left_parens(height):
    """Return an array containing each row of a large parenthesis
    used for pretty printing
    """
    a = ['⎧']
    for _ in range(height - 2):
        a.append('⎪')
    a.append('⎩')
    return a


def right_parens(height):
    """Return an array containing each row of a large parenthesis
    used for pretty printing
    """
    a = ['⎫']
    for _ in range(height - 2):
        a.append('⎪')
    a.append('⎭')
    return a


def bracket(a):
    """Append a left bracket for an array of lines"""
    if len(a) == 1:
        return '{' + a[0]
    if len(a) == 2:
        a.append('')
    if (len(a) % 2) == 0:
        upper = len(a) // 2 - 1
    else:
        upper = len(a) // 2
    a[0] = '⎧' + a[0]
    for i in range(1, upper):
        a[i] = '⎪' + a[i]
    a[upper] = '⎨' + a[upper]
    for i in range(upper + 1, len(a) - 1):
        a[i] = '⎪' + a[i]
    a[-1] = '⎩' + a[-1]
    return '\n'.join(a) + '\n'


def _character_to_superscript(ch: str) -> str:
    if ch == "0":
        return "⁰"
    elif ch == "1":
        return "¹"
    elif ch == "2":
        return "²"
    elif ch == "3":
        return "³"
    elif ch == "4":
        return "⁴"
    elif ch == "5":
        return "⁵"
    elif ch == "6":
        return "⁶"
    elif ch == "7":
        return "⁷"
    elif ch == "8":
        return "⁸"
    elif ch == "9":
        return "⁹"
    elif ch == "-":
        return "⁻"
    else:
        return ""


def int_to_superscript(x: int) -> str:
    return ''.join(_character_to_superscript(d) for d in str(x))


def itemize_strings(a):
    # Create an itemization string of a list of strings like:
    # ['a', 'b', 'c'] giving " one of 'a', 'b' or 'c'"
    if len(a) == 1:
        return repr(a[0])
    else:
        quoted = [repr(e) for e in a]
        return f" one of {', '.join(quoted[:-1])} or {quoted[-1]}"
