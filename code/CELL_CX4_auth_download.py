# =====================================================================
# CELL_CX4_auth_download.py  -- KGAT access-token path (Kaggle's new token)
# Upgrades kaggle (older versions don't support access tokens), takes the
# KGAT_ token via a HIDDEN prompt, writes it per Kaggle's official method,
# verifies login, and downloads train_transaction.csv. Run BEFORE CX4c.
# SECURITY: token entered hidden (getpass), written only to ~/.kaggle
# (chmod 600), never printed, never sent anywhere. Regenerate any token
# you have pasted in plaintext.
# =====================================================================
import os, glob, getpass, subprocess
print("upgrading kaggle client (needed for access tokens)...")
subprocess.run(["pip","-q","install","-U","kaggle"],check=False)

tok = getpass.getpass("Paste a FRESH KGAT_ token (input hidden): ").strip()
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
at = os.path.expanduser("~/.kaggle/access_token")
with open(at,"w") as f: f.write(tok)
os.chmod(at, 0o600)
os.environ["KAGGLE_API_TOKEN"] = tok
del tok  # drop from memory

# stage 1: verify auth (does NOT need competition rules)
r = subprocess.run(["kaggle","competitions","list"], capture_output=True, text=True)
if r.returncode == 0:
    print("AUTH OK.")
    # stage 2: download (needs rules accepted)
    d = subprocess.run(["kaggle","competitions","download","-c","ieee-fraud-detection",
                        "-f","train_transaction.csv","-p","."], capture_output=True, text=True)
    if d.returncode != 0:
        print("DOWNLOAD FAILED:", (d.stderr or d.stdout)[:400])
        if "403" in (d.stderr or "") or "forbidden" in (d.stderr or "").lower():
            print(">>> Accept rules then rerun: https://www.kaggle.com/competitions/ieee-fraud-detection/rules")
    for z in glob.glob("*.zip"):
        subprocess.run(["unzip","-o",z], check=False)
    print("train_transaction.csv present:", os.path.exists("train_transaction.csv"))
    if os.path.exists("train_transaction.csv"):
        print(">>> SUCCESS. Now run CELL_CX4c (it will detect the file and run the analysis).")
else:
    print("AUTH FAILED. Printed error (paste this back to me if it persists):")
    print((r.stderr or r.stdout)[:400])
    print(">>> Likely: token mistyped/expired. Regenerate a fresh KGAT_ token and rerun.")
