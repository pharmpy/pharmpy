from enum import Enum

class OptionType(Enum):
    BOOLEAN = 1
    VALUE = 2

class OptionDescriptionList:
    def __init__(self, struct):
        self.descriptions = []
        for s in struct:
            self.descriptions.append(OptionDescription(s))

    def match(self, name1, name2):
        for d in self.descriptions:
            if d.match(name1) and d.match(name2):
                return True
        return False

class OptionDescription:
    def __init__(self, struct):
        self.name = struct['name']
        self.type = struct['type']
        self.abbreviate = struct['abbreviate']

    def match(self, name):      # Could add abbreviation length here coming from the List
        name = name.upper()
        if self.abbreviate:
            return len(name) >= 3 and self.name.startswith(name)
        else:
            return self.name == name
