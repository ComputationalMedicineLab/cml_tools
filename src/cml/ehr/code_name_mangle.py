"""
Many EHR concept taxonomies use codes which are numeric or otherwise
non-semantic, but code names which are lengthy (sometimes very lengthy)
descriptions. The purpose of the functions in this module are to produce short,
easily legible labels from such code names.
"""
import functools
import re
import textwrap


UNITS_REGEX = re.compile(r'.*\[([^/]*)/?(.*)\]', flags=re.IGNORECASE)

UNITS_REPLACEMENT_HEAD = (
    ('#', '#'),
    ('Entitic mass', 'M'),
    ('Entitic volume', 'V'),
    ('Enzymatic activity', 'Activity'),
    ('Interpretation', 'Interp'),
    ('Length', 'L'),
    ('Log #', 'Log #'),
    ('Mass', 'M'),
    ('Mass Ratio', 'M/M'),
    ('Molar ratio', 'mol/mol'),
    ('Moles', 'mol'),
    ('Multiple of the median', 'Median mult'),
    ('Partial pressure', 'PP'),
    ('Percentile', r'%ile'),
    ('Presence', 'Presence'),
    ('Ratio', 'Ratio'),
    ('Susceptibility', 'Susc'),
    ('Time', 'T'),
    ('Titer', 'Titer'),
    ('Units', 'U'),
    ('Volume Fraction', 'VF'),
    ('Volume Rate', 'VR'),
    ('Z-score', 'Z'),
)

UNITS_REPLACEMENT_TAIL = (
    ('', None),
    ('area', 'A'),
    ('mass', 'M'),
    ('time', 'T'),
    ('volume', 'V'),
)

BLOOD_PANEL_REPLACEMENTS = (
    ('in Serum, Plasma, or Blood', 'in SPB'),
    ('in Serum or Plasma', 'in SP'),
    ('in Serum', 'in S'),
    ('in Plasma', 'in P'),
    ('in Blood', 'in B'),
    ('by automated count', 'by AC'),
)

_REP_HEAD = dict(UNITS_REPLACEMENT_HEAD)
_REP_TAIL = dict(UNITS_REPLACEMENT_TAIL)


@functools.cache
def mangle_description(description, width=80, placeholder='...'):
    """
    Reformat EHR concept code description strings, particularly for LOINC lab
    code names. Units and certain other specifiers are abbreviated and, if
    `width` is provided and greater than zero, the overall string is truncated
    to be at most `width` wide.

    Example:

    >>> mangle_description('Carbon dioxide, total [Moles/volume] in Blood')
    'Carbon dioxide, total [mol/V] in B'
    """
    # Step One: shorten units
    if (match := UNITS_REGEX.match(description)) is not None:
        head = match.group(1)
        replacement = _REP_HEAD.get(head, head)

        if (tail := match.group(2).lower()):
            tail = _REP_TAIL.get(tail, tail)
            replacement = f'{replacement}/{tail}'

        beginning = description[:match.span(1)[0]]
        end = description[match.span(2)[1]:]
        description = ''.join((beginning, replacement, end))

    # Step Two: mangle the "in Blood" style endings
    for phrase, replacement in BLOOD_PANEL_REPLACEMENTS:
        description = re.sub(phrase, replacement, description,
                             flags=re.IGNORECASE)

    # Step Three: remove excess whitespace and extraneous underscores
    description = re.sub(r'\s+', ' ', description)
    description = description.replace('_', ' ')

    # Step Four: shorten the string
    if width and width > 0:
        description = textwrap.shorten(description, width=width,
                                       placeholder=placeholder)
    return description
