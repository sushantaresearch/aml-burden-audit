# reconstruct_legal_form.py
# Reconstructs the legal-form (entity-type) EVALUATION grouping for AMLworld
# HI-Small, exactly as used in Section IV of the manuscript.
#
# The grouping labels each transaction by the ORIGINATOR account's registered
# legal form. It defines evaluation groups only; it is NEVER a model feature in
# either the thin or the rich detector.
#
# The exact entity field and its value-to-class mapping are the two named
# constants ENTITY_FIELD and MAPPING below. Everything else reproduces the
# canonical build: raw transaction parse, originator composite key, join to the
# accounts table, four-way coding, and a stable timestamp sort so the output is
# row-aligned to Small_HI_formatted_transactions.csv.
#
# Output: g_sorted_hismall.npy, one int8 code per formatted-transaction row,
# aligned to the time-sorted order. Codes:
#     0 = S  (Sole Proprietorship)
#     1 = M  (Partnership)
#     2 = L  (Corporation)
#     3 = other  (any account whose entity type is none of the above)
#
# Inputs (raw AMLworld HI-Small release files, from the Kaggle distribution):
#     HI-Small_Trans.csv
#     HI-Small_accounts.csv
#
# Usage:
#     python reconstruct_legal_form.py \
#         --trans HI-Small_Trans.csv \
#         --accounts HI-Small_accounts.csv \
#         --out g_sorted_hismall.npy
# Paths default to the file names above in the working directory.

import argparse
import numpy as np
import pandas as pd

# --- the two grouping constants referenced by the manuscript -----------------
ENTITY_FIELD = "Entity Name"   # column in HI-Small_accounts.csv
# Value-to-class mapping. The raw Entity Name is first reduced to its type
# prefix (text before " #"), then mapped. Anything not listed becomes "other".
MAPPING = {
    "Sole Proprietorship": "S",
    "Partnership":         "M",
    "Corporation":         "L",
}
CODE = {"S": 0, "M": 1, "L": 2, "other": 3}

# --- canonical integrity anchors (raw HI-Small; from the reference build) -----
N_ROWS_EXPECTED = 5_078_345
N_ACC_EXPECTED  = 515_088
COVERAGE_MIN    = 0.999
# Optional cross-check: on the locked test band (rows >= 4_214_445, the repo
# temporal split point), the four-way code counts are, for 0/1/2/3:
#     305225 / 304044 / 238426 / 16205   (sum 863900)
TEST_LO_EXPECTED = 4_214_445
TESTBAND_COUNTS_EXPECTED = (305225, 304044, 238426, 16205)


def build(trans_path, accounts_path, out_path):
    raw = pd.read_csv(trans_path, dtype=str)
    n = len(raw)
    assert n == N_ROWS_EXPECTED, f"row count {n} != expected {N_ROWS_EXPECTED}"

    # timestamp -> integer seconds from the first normalized day (minus 10s),
    # then a stable argsort giving the formatted-file row order.
    dtv = pd.to_datetime(raw["Timestamp"], format="%Y/%m/%d %H:%M")
    epoch = dtv.values.astype("int64") // 10**9
    first_ts = dtv.iloc[0].normalize().value // 10**9 - 10
    ts = (epoch - first_ts).astype("int64")
    order = np.argsort(ts, kind="stable")

    # optional account-count integrity check (confirms the raw file identity)
    ac = np.empty(2 * n, object)
    ac[0::2] = (raw["From Bank"] + raw["Account"]).values
    ac[1::2] = (raw["To Bank"] + raw["Account.1"]).values
    _, auniq = pd.factorize(ac)
    n_acc = len(auniq)
    assert n_acc == N_ACC_EXPECTED, f"account count {n_acc} != expected {N_ACC_EXPECTED}"

    # originator composite key: bank id (leading zeros stripped, empty -> "0")
    # joined to the account number (stripped, uppercased).
    nb = raw["From Bank"].str.strip().str.lstrip("0")
    nb = nb.where(nb != "", "0")
    skey = nb + "|" + raw["Account"].str.strip().str.upper()

    acc = pd.read_csv(accounts_path, dtype=str)
    acc.columns = [c.strip() for c in acc.columns]

    form = (acc[ENTITY_FIELD].str.split(" #").str[0].str.strip()
            .map(MAPPING).fillna("other"))
    ab = acc["Bank ID"].str.strip().str.lstrip("0")
    ab = ab.where(ab != "", "0")
    akey = ab + "|" + acc["Account Number"].str.strip().str.upper()

    g_row = skey.map(dict(zip(akey, form)))
    coverage = g_row.notna().mean()
    print(f"[join] originator coverage = {coverage:.4%}")
    assert coverage > COVERAGE_MIN, f"coverage {coverage:.4%} below floor"

    g_full = g_row.map(CODE).fillna(-1).astype("int8").values
    g_sorted = g_full[order]

    np.save(out_path, g_sorted)

    shares = {k: round(float((g_sorted == v).mean()), 4) for k, v in CODE.items()}
    print(f"[saved] {out_path}: {len(g_sorted):,} codes")
    print(f"[shares] S/M/L/other = {shares}")

    # optional test-band cross-check (non-fatal; prints PASS/DIFF)
    band = np.bincount(g_sorted[TEST_LO_EXPECTED:].clip(min=0), minlength=4)[:4]
    got = tuple(int(x) for x in band)
    status = "PASS" if got == TESTBAND_COUNTS_EXPECTED else "DIFF"
    print(f"[test-band 0/1/2/3] got {got} expected {TESTBAND_COUNTS_EXPECTED} -> {status}")

    # optional alignment check against the formatted file, if present
    return g_sorted, order


def verify_alignment(order, formatted_path):
    try:
        fm = pd.read_csv(formatted_path, usecols=["Is Laundering"])
    except (FileNotFoundError, ValueError):
        return
    # rebuild y in raw order is not needed here; the formatted file is already
    # in sorted order, so its length must equal the number of rows we sorted.
    if len(fm) == len(order):
        print(f"[align] {formatted_path}: row count matches ({len(fm):,}).")
    else:
        print(f"[align] WARNING: {formatted_path} has {len(fm):,} rows, "
              f"expected {len(order):,}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trans", default="HI-Small_Trans.csv")
    ap.add_argument("--accounts", default="HI-Small_accounts.csv")
    ap.add_argument("--out", default="g_sorted_hismall.npy")
    ap.add_argument("--formatted", default="Small_HI_formatted_transactions.csv",
                    help="optional; used only for a row-count alignment check")
    args = ap.parse_args()
    _, order = build(args.trans, args.accounts, args.out)
    verify_alignment(order, args.formatted)


if __name__ == "__main__":
    main()
