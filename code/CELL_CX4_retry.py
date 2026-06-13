# =====================================================================
# CELL_CX4_retry.py -- ONE clean retry: force-reinstall kaggle, capture the
# REAL import error (earlier traceback was truncated), reuse the existing
# ~/.kaggle/access_token (no token re-entry), verify, download.
# Optional: Runtime > Restart session before running, to clear stale state.
# Everything goes through subprocesses, so a clean Python process is used for
# the import check and the download.
# =====================================================================
import os, glob, subprocess, sys

print(">>> Step 1: clean force-reinstall of kaggle (no cache) ...")
subprocess.run([sys.executable,"-m","pip","uninstall","-y","kaggle","-q"])
ins=subprocess.run([sys.executable,"-m","pip","install","--force-reinstall","--no-cache-dir","-q","kaggle"],
                   capture_output=True,text=True)
if ins.returncode!=0: print("pip note:",(ins.stderr or ins.stdout)[-500:])

print(">>> Step 2: import check in a clean subprocess (captures the real error) ...")
chk=subprocess.run([sys.executable,"-c","import kaggle,sys;print('IMPORT_OK',getattr(kaggle,'__version__','?'))"],
                   capture_output=True,text=True)
print(chk.stdout.strip())
IMPORT_OK = chk.returncode==0 and "IMPORT_OK" in chk.stdout
if not IMPORT_OK:
    print("---- FULL IMPORT TRACEBACK (copy this whole block back to me) ----")
    print(chk.stderr[-1500:])
    print("-----------------------------------------------------------------")

if IMPORT_OK:
    at=os.path.expanduser("~/.kaggle/access_token")
    if os.path.exists(at):
        os.environ["KAGGLE_API_TOKEN"]=open(at).read().strip()
        print(">>> Step 3: reusing existing access_token (no re-entry).")
    else:
        import getpass
        tok=getpass.getpass("Paste KGAT_ token (hidden): ").strip()
        os.makedirs(os.path.expanduser("~/.kaggle"),exist_ok=True)
        open(at,"w").write(tok); os.chmod(at,0o600); os.environ["KAGGLE_API_TOKEN"]=tok; del tok
        print(">>> Step 3: token stored.")

    print(">>> Step 4: verify auth ...")
    v=subprocess.run(["kaggle","competitions","list"],capture_output=True,text=True)
    if v.returncode==0:
        print("AUTH OK. Downloading train_transaction.csv ...")
        d=subprocess.run(["kaggle","competitions","download","-c","ieee-fraud-detection",
                          "-f","train_transaction.csv","-p","."],capture_output=True,text=True)
        if d.returncode!=0:
            print("DOWNLOAD FAILED:",(d.stderr or d.stdout)[:400])
            if "403" in (d.stderr or "") or "forbidden" in (d.stderr or "").lower():
                print(">>> Accept rules then rerun: https://www.kaggle.com/competitions/ieee-fraud-detection/rules")
        for z in glob.glob("*.zip"): subprocess.run(["unzip","-o",z],check=False)
        ok=os.path.exists("train_transaction.csv"); print("train_transaction.csv present:",ok)
        if ok: print(">>> SUCCESS. Now run CELL_CX4c -- it detects the file and runs the analysis.")
    else:
        print("AUTH FAILED. Error (paste back):"); print((v.stderr or v.stdout)[:400])
        print(">>> If this says credentials/401, this kaggle version does not accept KGAT tokens;")
        print(">>> paste it back and I will pin a KGAT-capable version or switch to a no-Kaggle dataset.")
