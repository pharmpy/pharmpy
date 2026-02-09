import re
from datetime import datetime
from typing import Optional

import dateutil


def parse_datestamp(row: str, row_next: Optional[str] = None) -> Optional[datetime]:
    weekday_month_en = re.compile(
        r'^\s*(Sun|Mon|Tue|Wed|Thu|Fri|Sat)'
        r'\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'  # Month
        r'\s+(\d+)'  # Day
        r'\s+.*'
        r'\s+(\d{4})'  # Year
    )
    weekday_month_sv = re.compile(
        r'^\s*(mån|tis|ons|tor|fre|lör|sön)'
        r'\s+(\d+)'  # Day
        r'\s+(jan|feb|mar|apr|maj|jun|jul|aug|sep|okt|nov|dec)'  # Month
        r'\s+(\d+)'  # Year
    )
    day_month_year = re.compile(r'^(\d{2})/(\d{2})/(\d{4})\s*$')  # dd/mm/yyyy
    year_month_day = re.compile(r'^(\d{4})-\d{2}-\d{2}\s*$')  # yyyy/mm/dd
    timestamp = re.compile(r'([0-9]{2}:[0-9]{2}:[0-9]{2} (AM|PM)?)')

    month_no = {
        'JAN': 1,
        'FEB': 2,
        'MAR': 3,
        'APR': 4,
        'MAY': 5,
        'JUN': 6,
        'JUL': 7,
        'AUG': 8,
        'SEP': 9,
        'OCT': 10,
        'NOV': 11,
        'DEC': 12,
    }
    month_trans = {'MAJ': 'MAY', 'OKT': 'OCT'}

    def _dmy(row):
        if match_en := weekday_month_en.match(row):
            _, month, day, year = match_en.groups()
            return day, month, year

        elif match_sv := weekday_month_sv.match(row):
            _, day, month, year = match_sv.groups()
            return day, month, year

        return None

    dmy = _dmy(row)
    if dmy is not None:
        day, month, year = dmy
        try:
            month = month_no[month.upper()]
        except KeyError:
            month_en = month_trans[month.upper()]
            month = month_no[month_en.upper()]

        date = datetime(int(year), int(month), int(day))

        match = timestamp.search(row)
        if match is None:
            return date

        time_str = match.groups()[0]
        time = dateutil.parser.parse(time_str).time()

    elif (match_day_first := day_month_year.match(row)) or year_month_day.match(row):
        date = dateutil.parser.parse(row, dayfirst=bool(match_day_first))

        if row_next is None:
            return date

        time = dateutil.parser.parse(row_next).time()
    else:
        return None

    combined = datetime.combine(date, time)
    return combined
