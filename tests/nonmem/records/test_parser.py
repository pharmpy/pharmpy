
import pytest


def test_problem_record(parser, nm_csv_read, str_repr):
    print()
    problem_parser = parser.ProblemRecordParser
    data = nm_csv_read('problem_record.csv', ['input', 'text', 'comment'])

    for row in data:
        for i, line in enumerate(row.input.splitlines()):
            print('line %-2d %s' % (i, str_repr(line)))
        parser = problem_parser(row[0])
        print(parser, end='\n\n')
        assert parser.root
        assert parser.root.text.TEXT == row.text
        if row.comment:
            comments = parser.root.all('comment')
            for out, ref in zip(comments, row.comment):
                assert out.COMMENT == ref
