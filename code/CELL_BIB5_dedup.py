# ============================================================
# CELL BIB-5 -- DEDUP + CLEAN references_v3_merged.bib
# Resolves the 6 duplicate clusters (keeps pre-existing in-body keys with
# corrected bodies, drops my appended duplicates), normalizes all page
# ranges to "--", and reports non-ASCII + re-scans. Writes a NEW file
# references_v3_clean.bib (v3_merged left untouched). CPU ~10 sec.
# Paste the full printed report.
# ============================================================
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
import glob, os, re

cands = sorted(glob.glob('/content/drive/MyDrive/**/references_v3_merged.bib', recursive=True))
assert cands, '[FAIL] references_v3_merged.bib not found'
SRC = cands[0]
OUTDIR = os.path.dirname(SRC)
OUT = os.path.join(OUTDIR, 'references_v3_clean.bib')
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

# ---- corrected bodies for the 6 KEPT keys ----
FINAL = {
'dworkroth2014': r'''@article{dworkroth2014, title={The Algorithmic Foundations of Differential Privacy}, volume={9}, number={3-4}, ISSN={1551-3068}, DOI={10.1561/0400000042}, url={http://dx.doi.org/10.1561/0400000042}, journal={Foundations and Trends in Theoretical Computer Science}, author={Dwork, Cynthia and Roth, Aaron}, year={2014}, pages={211--487} }''',
'reite2023': r'''@article{reite2023, title={Changes in credit score, transaction volume, customer characteristics, and the probability of detecting suspicious transactions}, volume={26}, number={6}, ISSN={1368-5201}, DOI={10.1108/JMLC-06-2022-0087}, url={http://dx.doi.org/10.1108/jmlc-06-2022-0087}, journal={Journal of Money Laundering Control}, publisher={Emerald}, author={Reite, Endre J. and Oust, Are and Bang, Rebecca Margareta and Maurstad, Stine}, year={2023}, pages={1165--1178} }''',
'reite2024': r'''@article{reite2024, author={Reite, Endre Jo and Karlsen, Johan and Westgaard, Elias Grefstad}, title={Improving client risk classification with machine learning to increase anti-money laundering detection efficiency}, journal={Journal of Money Laundering Control}, volume={28}, number={1}, pages={93--107}, year={2025}, doi={10.1108/JMLC-03-2024-0040}, note={First published online 15 August 2024}}''',
'ye2026fairgse': r'''@inproceedings{ye2026fairgse, title={{FairGSE}: Fairness-Aware Graph Neural Network Without High False Positive Rates}, author={Ye, Zhenqiang and Lu, Jinjie and Gu, Tianlong and Hao, Fengrui and Wang, Xuemin}, booktitle={Proceedings of the {AAAI} Conference on Artificial Intelligence}, volume={40}, number={19}, pages={16163--16171}, year={2026}, doi={10.1609/aaai.v40i19.38652}}''',
'johannessen2023dnb': r'''@article{johannessen2023dnb, author={Johannessen, Fredrik and Jullum, Martin}, title={Finding Money Launderers Using Heterogeneous Graph Neural Networks}, journal={The Journal of Finance and Data Science}, volume={11}, pages={100175}, year={2025}, doi={10.1016/j.jfds.2025.100175}}''',
'jesus2022baf': r'''@inproceedings{jesus2022baf, author={Jesus, S{\'e}rgio and Pombal, Jos{\'e} and Alves, Duarte and Cruz, Andr{\'e} F. and Saleiro, Pedro and Ribeiro, Rita P. and Gama, Jo{\~a}o and Bizarro, Pedro}, title={Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for {ML} Evaluation}, booktitle={Advances in Neural Information Processing Systems 35 ({NeurIPS} 2022) Datasets and Benchmarks Track}, year={2022}}''',
}
DROP = {'dwork2014dp', 'reite2023changes', 'reite2025improving',
        'ye2025fairgse', 'johannessen2025finding', 'jesus2022turning'}

def fix_pages(block):
    m = re.search(r'pages\s*=\s*\{([^}]*)\}', block, re.I)
    if not m: return block
    content = m.group(1)
    mm = re.match(r'^\s*(\d+)\s*\D+?\s*(\d+)\s*$', content)
    if not mm: return block
    newc = mm.group(1) + '--' + mm.group(2)
    return block[:m.start(1)] + newc + block[m.end(1):]

ents = parse_bib(raw)
print('[v3_merged entries]', len(ents))

out_blocks, kept_final, dropped = [], [], []
for e in ents:
    k = e['key']
    if k in DROP:
        dropped.append(k); continue
    block = FINAL[k] if k in FINAL else e['block']
    if k in FINAL: kept_final.append(k)
    out_blocks.append(fix_pages(block).strip())

header = ('% references_v3_clean.bib -- deduped + page-normalized from '
          'references_v3_merged.bib\n'
          '% 6 duplicate clusters collapsed onto pre-existing in-body keys; '
          'see BIB_DEDUP_v3_decisions.txt\n\n')
clean = header + '\n\n'.join(out_blocks) + '\n'
open(OUT, 'w', encoding='utf-8').write(clean)

# ---- re-scan ----
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
        return block[k+1:block.find('"', k+1)]
    m2 = re.match(r'([^,\n]+)', block[k:])
    return m2.group(1).strip() if m2 else None

def norm_title(t):
    if not t: return ''
    t = re.sub(r'[{}\\]', '', t)
    return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', t.lower()).split())

ents2 = parse_bib(clean)
keys2 = [e['key'] for e in ents2]
print('\n[REKEYED bodies applied]', sorted(kept_final))
print('[DROPPED]', sorted(dropped))
print('[clean entries]', len(ents2))

dm = {}
for e in ents2:
    d = field(e['block'], 'doi')
    if d: dm.setdefault(d.strip().lower(), []).append(e['key'])
doidup = {d: ks for d, ks in dm.items() if len(ks) > 1}
print('[DOI dups]', doidup if doidup else 'none')

tm = {}
for e in ents2:
    nt = norm_title(field(e['block'], 'title'))
    if nt: tm.setdefault(nt, []).append(e['key'])
titledup = {t: ks for t, ks in tm.items() if len(ks) > 1}
print('[title dups]', titledup if titledup else 'none')

print('[em-dash U+2014]', clean.count('\u2014'))

dupkeys = sorted({k for k in keys2 if keys2.count(k) > 1})
print('[duplicate keys]', dupkeys if dupkeys else 'none')

# pages sanity: any pages field still holding non-digit/non "--"
bad_pages = []
for e in ents2:
    p = field(e['block'], 'pages')
    if p and not re.match(r'^\s*\d+(\s*--\s*\d+)?\s*$', p):
        bad_pages.append((e['key'], p))
print('[non-normalized pages]', bad_pages if bad_pages else 'none')

# non-ASCII health report (review only)
print('\n===== NON-ASCII CHARACTERS (review; legitimate accents OK) =====')
seen = 0
for e in ents2:
    for mch in re.finditer(r'[^\x00-\x7f]', e['block']):
        i = mch.start(); c = mch.group(0)
        ctx = e['block'][max(0, i-15):i+15].replace('\n', ' ')
        print('%-22s U+%04X %r | ...%s...' % (e['key'], ord(c), c, ctx))
        seen += 1
print('(total non-ASCII chars: %d)' % seen)
print('\n[saved]', OUT)
print('[DONE BIB-5 DEDUP]')
