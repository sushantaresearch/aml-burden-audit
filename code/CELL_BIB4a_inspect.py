# ============================================================
# CELL BIB-4a -- INSPECT ONLY (no mutation). ~5 sec, CPU.
# Lists all .bib on the drive, dumps the 10 entries in the 5
# suspected duplicate pairs so bodies can be compared, runs a
# global title-near-duplicate scan, and greps every .tex for
# \cite of the target keys. Paste the FULL output back.
# ============================================================
import glob, os, re
base = '/content/drive/MyDrive'

print('== all .bib files on drive ==')
for p in sorted(glob.glob(base + '/**/*.bib', recursive=True)):
    print('   ', p, os.path.getsize(p), 'bytes')

v3s = glob.glob(base + '/**/references_v3_merged.bib', recursive=True)
assert v3s, '[FAIL] references_v3_merged.bib not found'
V3 = sorted(v3s)[0]
print('\n[v3 used]', V3)
raw = open(V3, encoding='utf-8', errors='replace').read()

def parse_bib(text):
    out, i, n = [], 0, len(text)
    while True:
        m = re.search(r'@(\w+)\s*\{', text[i:])
        if not m: break
        s = i + m.start(); j = i + m.end(); d = 1
        while j < n and d > 0:
            if text[j] == '{': d += 1
            elif text[j] == '}': d -= 1
            j += 1
        b = text[s:j]; km = re.match(r'@\w+\s*\{\s*([^,\s]+)', b)
        out.append((km.group(1) if km else None, b)); i = j
    return out

E = parse_bib(raw); D = dict(E)
print('[v3 entries]', len(E))

pairs = [('dworkroth2014', 'dwork2014dp'),
         ('reite2023', 'reite2023changes'),
         ('reite2024', 'reite2025improving'),
         ('johannessen2023dnb', 'johannessen2025finding'),
         ('ye2026fairgse', 'ye2025fairgse')]

for a, b in pairs:
    print('\n' + '=' * 72)
    for k in (a, b):
        print('---- ' + k + ' ----')
        print(D.get(k, '   <<MISSING FROM v3>>'))

def norm_title(b):
    m = re.search(r'title\s*=\s*\{(.+?)\}\s*,', b, re.S | re.I)
    if not m: return ''
    t = m.group(1).lower(); t = re.sub(r'[^a-z0-9]+', ' ', t)
    return ' '.join(t.split())

print('\n== global title-near-duplicate scan (shared first 8 title words) ==')
buckets = {}
for k, b in E:
    pref = ' '.join(norm_title(b).split()[:8])
    if pref: buckets.setdefault(pref, []).append(k)
flagged = False
for pref, ks in buckets.items():
    if len(ks) > 1:
        flagged = True; print('   ', ks, '::', pref)
if not flagged: print('    none')

print('\n== \\cite occurrences of target keys in .tex files ==')
targets = set(sum([list(p) for p in pairs], []))
any_tex = False
for p in sorted(glob.glob(base + '/**/*.tex', recursive=True)):
    any_tex = True
    try: tx = open(p, encoding='utf-8', errors='replace').read()
    except Exception as e: print('   [skip]', p, e); continue
    hits = sorted({k for k in targets
                   if re.search(r'[\{,]\s*' + re.escape(k) + r'\s*[\},]', tx)})
    if hits: print('   ', os.path.basename(p), '->', hits)
if not any_tex: print('    (no .tex files found on drive)')

print('\n[DONE BIB-4a inspect]')
