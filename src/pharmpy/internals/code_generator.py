class CodeGenerator:
    def __init__(self):
        self.indent_level = 0
        self.lines = []

    def indent(self):
        self.indent_level += 4

    def dedent(self):
        self.indent_level -= 4

    def add(self, line):
        self.lines.append(f'{" " * self.indent_level}{line}')

    def empty_line(self):
        self.lines.append('')

    def __str__(self):
        return '\n'.join(self.lines)
