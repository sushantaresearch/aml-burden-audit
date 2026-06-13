# ============================================================
# CELL BIB-4 -- AUDIT references_v3_merged.bib for duplicate works
# READ-ONLY. Finds duplicate clusters by (a) shared DOI and
# (b) identical / near-identical title, and prints the FULL body
# of every entry involved so the exact merge can be adjudicated.
# Modifies NOTHING. CPU, ~5 sec. Paste the entire printed output.
# ============================================================
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
import glob, os, re

cands = sorted(glob.glob('/content/drive/MyDrive/**/references_v3_merged.bib', recursive=True))
assert cands, '[FAIL] references_v3_merged.bib not found'
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
        block = text[start:j]
        km = re.match(r'@\w+\s*\{\s*([^,\s]+)\s*,', block)
        entries.append({'key': km.group(1) if km else None, 'block': block})
        i = j
    return entries

def field(block, name):
    m = re.search(r'\b' + name + r'\s*=\s*', block, re.I)
    if not m: return None
    k = m.end()
    if k < len(block) and block[k] == '{':
        depth, j = 1, k + 1
        while j < len(block) and depth > 0:
            if block[j] == '{': depth += 1
            elif block[j] == '}': depth -= 1
            j += 1
        return block[k+1:j-1]
    if k < len(block) and block[k] == '"':
        j = block.find('"', k + 1)
        return block[k+1:j]
    m2 = re.match(r'([^,\n]+)', block[k:])
    return m2.group(1).strip() if m2 else None

def norm_title(t):
    if not t: return ''
    t = re.sub(r'[{}\\]', '', t)
    t = re.sub(r'[^a-z0-9 ]', ' ', t.lower())
    return ' '.join(t.split())

ents = parse_bib(raw)
print('[entries]', len(ents))
bodies = {e['key']: e['block'] for e in ents}

# ---- DOI clusters ----
doi_map = {}
for e in ents:
    d = field(e['block'], 'doi')
    if d:
        doi_map.setdefault(d.strip().lower(), []).append(e['key'])
doi_dups = {d: ks for d, ks in doi_map.items() if len(ks) > 1}

doi_pairs = set()
for ks in doi_dups.values():
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            doi_pairs.add(frozenset((ks[a], ks[b])))

# ---- exact normalized-title clusters ----
tmap = {}
for e in ents:
    nt = norm_title(field(e['block'], 'title'))
    if nt:
        tmap.setdefault(nt, []).append(e['key'])
title_exact = {t: ks for t, ks in tmap.items() if len(ks) > 1}

# ---- title near-duplicates (Jaccard >= 0.6), excluding DOI-clustered pairs ----
titles = [(e['key'], set(norm_title(field(e['block'], 'title')).split())) for e in ents]
near = []
for a in range(len(titles)):
    for b in range(a + 1, len(titles)):
        ka, sa = titles[a]; kb, sb = titles[b]
        if not sa or not sb: continue
        if frozenset((ka, kb)) in doi_pairs: continue
        jac = len(sa & sb) / len(sa | sb)
        if jac >= 0.6:
            near.append((round(jac, 2), ka, kb))

print('\n===== DOI-SHARED CLUSTERS (%d) =====' % len(doi_dups))
for d, ks in doi_dups.items():
    print('\n>>> DOI', d, '->', ks)
    for k in ks:
        print('-' * 60); print(bodies[k].strip())

print('\n===== EXACT-NORMALIZED-TITLE CLUSTERS =====')
shown = 0
for t, ks in title_exact.items():
    if all(frozenset((ks[i], ks[j])) in doi_pairs
           for i in range(len(ks)) for j in range(i + 1, len(ks))):
        continue  # already fully explained by DOI
    shown += 1
    print('\n>>> TITLE "%s" -> %s' % (t[:70], ks))
    for k in ks:
        print('-' * 60); print(bodies[k].strip())
print('(title-only clusters shown: %d)' % shown)

print('\n===== TITLE NEAR-DUPLICATES (Jaccard >= 0.6, %d) =====' % len(near))
for jac, ka, kb in sorted(near, reverse=True):
    print('\n>>> Jaccard %.2f : %s  <>  %s' % (jac, ka, kb))
    print('-' * 60); print(bodies[ka].strip())
    print('-' * 60); print(bodies[kb].strip())

print('\n[DONE BIB-4 AUDIT]')
