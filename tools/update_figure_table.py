"""Regenerate the figure -> script table of README.md from the scripts' docstrings.

The table sits between the <!-- figure-table:start --> and
<!-- figure-table:end --> markers; the first docstring line of each script in
figures/ gives its description.

Usage (repo root): python tools/update_figure_table.py
"""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
START, END = '<!-- figure-table:start -->', '<!-- figure-table:end -->'
SECTIONS = [
    ('figure_1', 'Figure 1'),
    ('figure_2', 'Figure 2'),
    ('figure_3', 'Figure 3'),
    ('figure_4', 'Figure 4'),
    ('supp_1', 'Figure 1 – supplement'),
    ('supp_2', 'Figure 2 – supplement'),
    ('supp_3', 'Figure 3 – supplement'),
    ('supp_4', 'Figure 4 – supplement'),
    ('revisions', 'Revision analyses (in progress)'),
]


def describe(path):
    doc = ast.get_docstring(ast.parse(path.read_text())) or ''
    first = doc.strip().split('\n\n')[0].replace('\n', ' ')
    # Drop a leading "Figure 1b:" / "Supplementary Figure 3m:" label.
    if ':' in first[:40]:
        first = first.split(':', 1)[1].strip()
    first = first.rstrip('.')
    return first[:1].upper() + first[1:]


def panel_label(stem):
    """figure_3h_j -> '3h–j' (range), supp_3c_d_g_h -> 'S3c, d, g, h'."""
    prefix = 'S' if stem.startswith('supp_') else ''
    number_and_letters = stem.split('_', 1)[1].split('_')
    first, letters = number_and_letters[0], number_and_letters[1:]
    if len(letters) == 1 and ord(letters[0]) - ord(first[-1]) > 1:
        return f'{prefix}{first}–{letters[0]}'
    return prefix + ', '.join([first] + letters)


def table():
    rows = ['| Panel(s) | Script | Content |', '|---|---|---|']
    for folder, title in SECTIONS:
        scripts = sorted((REPO / 'figures' / folder).glob('*.py'))
        if not scripts:
            continue
        rows.append(f'| **{title}** | | |')
        for s in scripts:
            panel = panel_label(s.stem) if folder != 'revisions' else '—'
            rows.append(f'| {panel} | [`{s.relative_to(REPO)}`]({s.relative_to(REPO)}) | {describe(s)} |')
    return '\n'.join(rows)


if __name__ == '__main__':
    readme = REPO / 'README.md'
    text = readme.read_text()
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f'{head}{START}\n{table()}\n{END}{tail}')
    print('README figure table updated.')
