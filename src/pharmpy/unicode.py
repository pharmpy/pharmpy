import math

mathematical_script_capital_n = '𝒩 '


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
                    length = e.minimum_width
                    if length > maxlen:
                        maxlen = length
            widths.append(maxlen)
        return widths

    def _row_height(self, row):
        maxheight = 0
        for col in range(self.ncols):
            e = self._grid[row][col]
            if e:
                n = e.minimum_height
                if n > maxheight:
                    maxheight = n
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
        self.text = text

    @property
    def minimum_width(self):
        return len(self.text) + 2

    @property
    def minimum_height(self):
        return 3

    def lines(self, width):
        before, after = _pad_on_sides(width - self.minimum_width)
        upper = '┌' + '─' * (width - 2) + '┐'
        mid = '│' + before + self.text + after + '│'
        lower = '└' + '─' * (width - 2) + '┘'
        return [upper, mid, lower]


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
