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
        
    def remove(self, line):
        # Remove the first instance exactly equal to input
        for n, code_line in enumerate(self.lines):
            if line in code_line:
                self.lines.pop(n)
                break

    def __str__(self):
        return '\n'.join(self.lines)
