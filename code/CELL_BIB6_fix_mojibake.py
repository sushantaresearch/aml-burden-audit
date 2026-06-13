# ============================================================
# CELL BIB-6 -- fix the TWO mojibake strings in references_v3_clean.bib
# Surgical, scoped by entry key. Leaves all legitimate accents untouched.
# Overwrites references_v3_clean.bib in place. CPU ~3 sec.
# Paste the full printed report.
# ============================================================
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
import glob, os, re

cands = sorted(glob.glob('/content/drive/MyDrive/**/references_v3_clean.bib', recursive=True))
assert cands, '[FAIL] references_v3_clean.bib not found'
SRC = cands[0]
print('[src]', SRC)
raw = open(SRC, encoding='utf-8', errors='replace').read()

def parse_bib(text):
    entries, i, n = [], 0, len(text)
    while True:
        m = re.search(r'@(\w+)\s*\{', text[i:])
        if not m: break
        start = i + m.start(); j = i + m.end(); depth = 1
        while j < n and depth > 0:
            if text[j] == '{': depth += 1
            elif text[j] == '}': depth -= 1
            j += 1
        entries.append({'start': start, 'end': j, 'block': text[start:j]})
        km = re.match(r'@\w+\s*\{\s*([^,\s]+)\s*,', text[start:j])
        entries[-1]['key'] = km.group(1) if km else None
        i = j
    return entries

# scoped replacements: key -> list of (old_substr, new_substr)
FIXES = {
 'jensen2023synthaml': [('J\u00c3\u00b8rgensen', 'J{\\o}rgensen')],
 'kwak2024tables':     [('\u201a\u00c4\u00ea', '-')],
}

ents = parse_bib(raw)
report = []
# rebuild text with replacements applied inside the targeted blocks only
pieces, cursor = [], 0
for e in ents:
    pieces.append(raw[cursor:e['start']])  # inter-entry text
    blk = e['block']
    if e['key'] in FIXES:
        for old, new in FIXES[e['key']]:
            c = blk.count(old)
            if c:
                blk = blk.replace(old, new)
            report.append((e['key'], repr(old), repr(new), c))
    pieces.append(blk)
    cursor = e['end']
pieces.append(raw[cursor:])
fixed = ''.join(pieces)

open(SRC, 'w', encoding='utf-8').write(fixed)

print('\n[REPLACEMENTS]')
for k, old, new, c in report:
    print('  %-22s %s -> %s  (hits: %d)' % (k, old, new, c))

# verify the two targets are clean now, and re-report all non-ASCII
ents2 = parse_bib(fixed)
print('\n[entries]', len(ents2))
print('[em-dash U+2014]', fixed.count('\u2014'))

def show_nonascii(block, key):
    out = []
    for mch in re.finditer(r'[^\x00-\x7f]', block):
        i = mch.start(); c = mch.group(0)
        ctx = block[max(0, i-15):i+15].replace('\n', ' ')
        out.append('%-22s U+%04X %r | ...%s...' % (key, ord(c), c, ctx))
    return out

# confirm targets clean
for e in ents2:
    if e['key'] in FIXES:
        na = show_nonascii(e['block'], e['key'])
        print('\n[target %s] non-ASCII now: %s' % (e['key'], 'none' if not na else ''))
        for line in na: print('   ', line)

print('\n===== ALL REMAINING NON-ASCII (should be legit accents/curly quotes) =====')
total = 0
for e in ents2:
    for line in show_nonascii(e['block'], e['key']):
        print(line); total += 1
print('(total remaining non-ASCII: %d)' % total)
print('\n[saved]', SRC)
print('[DONE BIB-6]')
