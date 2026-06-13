# aml-burden-audit

Code and result artifacts for the manuscript:

**"Who the Screen Flags: Auditing and Mitigating Structure-Conditioned
False-Positive Burden in Anti-Money-Laundering Screening"**
Submitted to IEEE Transactions on Computational Social Systems (2026).

Sole author: Sushanta Paul (ORCID: 0009-0007-8071-6971), National Board of
Revenue, Government of Bangladesh. Contact: sushanta.researcher@gmail.com

Archived snapshot: [Zenodo DOI inserted after v1.0 release]

## What this repository contains

Every table in the paper and its Supplementary Material is backed by a
result file in `results/`, and every number traces to a file per the
provenance rule used during manuscript engineering ("no number without a
source"). The audit pipeline, detector configurations, the four graph-run
seeds, the exact temporal split boundaries, and the Multi-GNN porting notes
are in `code/`, as promised in the paper's Reproducibility Statement.

## Layout

- `README.md` (this file)
- `LICENSE` (MIT, applies to code; result files are CC-BY-4.0, see below)
- `references_v4_final.bib` (machine-readable bibliography, DOI-verified)
- `code/` audit and training scripts (Colab cells exported as .py),
  detector hyperparameters, seeds, split boundaries, Multi-GNN porting notes
- `results/` small result files backing each table (manifest below)
- `figures/` publication figure PDFs (fig_bh_forest.pdf,
  fig_dose_response_v2.pdf)

Large artifacts (per-transaction score files of 20-77 MB, model checkpoint
.tar files, centrality .npz, edge-weight .npy) are attached to the GitHub
v1.0 RELEASE as release assets and mirrored in the Zenodo deposit rather
than committed to the repository, to keep clones light.

## Results manifest (table -> file)

Main article:
- Table I (burden ladder, naive intervals, BH family): bh_family_results.csv
- Table I (account-clustered intervals, both frames, design effects):
  cluster_bootstrap_both_frames_hismall.csv
- Four-seed graph band (Sec. V text): amlworld_gnn_multiseed_summary.csv,
  amlworld_gnn_multiseed_ladder.csv
- Table II (degree-matched gap decomposition): degree_explains_gap_table.csv
- Table III (SAML-D channel decomposition): samld_mechanism_decomposition.csv,
  samld_mechanism_matched_gaps_pp.csv
- SAML-D burden and permutation p-values (Sec. VII):
  samld_permutation_nulls.csv; AMLworld counterpart:
  amlworld_permutation_nulls.csv
- Budget sweep (Sec. VII): amlworld_R_budget_sweep.csv
- Entity/legal-form burden base: entity_burden_hismall.csv

Supplementary Material:
- S.2 / S.3 (degree deciles, 10% and 5%): degree_fpr_decile_10pct.csv,
  degree_fpr_decile_5pct.csv
- S.4 (currency cohorts): currency_nontransfer_R.csv
- S.5 / S.6 (reweighting dose and per-seed replicates):
  amlworld_gnn_4c_w1.5_ladder.csv, amlworld_gnn_4c_w2.5_ladder.csv,
  amlworld_gnn_4c_w4.0_ladder.csv, amlworld_gnn_4c_w2.5_seed3_ladder.csv,
  amlworld_gnn_4c_w2.5_seed4_ladder.csv
- Post-processing outcomes (Sec. VIII prose): mitigation_joint_results.csv,
  mitigation_results.csv
- S.7 (privacy clipping): privacy_clipping_table.csv
- S.8 / S.9 (customs per-origin audit and origin-tier contrast):
  customs_perorigin_fraud.csv, customs_perorigin_crit2.csv (per-origin
  one-vs-rest burden ratios with importer-clustered bootstrap CIs and BH,
  from CELL_CX3), customs_nonoecd_oecd.csv (non-OECD vs OECD contrast
  across budgets, clustered vs naive with design effects), and
  customs_burden_results.csv (strict UN-LDC vs OECD table from CELL_CX2;
  the LDC tier is reported as not estimable at n=42)
- Robustness extras: amlworld_xgb_hp_robustness.csv,
  amlworld_fairness_toolkit_comparison.csv, gnn_calibration_drift.csv,
  amlworld_centrality_panel.csv, samld_support_sensitivity.csv,
  three_way_fairness_table.csv

Release assets (not in repo):
- Per-transaction scores: amlworld_gnn_{val,test}_scores_seed{2,3,4}.csv,
  amlworld_gnn_{val,test}_scores_4c_w{1.5,2.5,4.0}*.csv,
  samld_{thin,rich}_xgb_{val,test}_scores.csv,
  samld_gnn_{val,test}_scores*.csv
- Checkpoints: checkpoint_seed{2,3,4}.tar, checkpoint_True.tar,
  checkpoint_4c_w{1.5,2.5,4.0}_seed*.tar
- Graph inputs: amlworld_centralities.npz, train_edge_weights_kc.npy,
  samld_accounts.csv, samld_account_degree.csv
- Split and encoding records: samld_split_days.json,
  samld_encoding_maps.json, samld_dataset_stats.json, samld_gnn_train_log.txt

## Data sources (not redistributed)

Raw benchmarks are NOT redistributed here. Obtain them from their original
sources: AMLworld HI-Small (IBM, via Kaggle), SAML-D (via Kaggle), and the
public customs import-declarations benchmark of Jeong et al. Score files in
the release contain model outputs keyed to transaction indices.
[TODO: confirm score CSVs carry no raw source columns beyond index, label,
and score before publishing the release; if they do, either drop those
columns or confirm the benchmark license permits redistribution.]

## Reproduce

1. Create the environment: the cells were run on Google Colab GPU runtimes
   (2026 images; Python 3 with numpy, pandas, scikit-learn, and matplotlib
   preinstalled). Additionally install torch and torch-geometric matching
   the runtime CUDA version, and clone the IBM Multi-GNN repository (see
   code/README_data_paths.txt). No exact version pins were recorded;
   versions follow the Colab image defaults at run time.
2. Download the raw benchmarks from their original sources (links above)
   and place them at the paths described in code/README_data_paths.txt
   (included).
3. Run the cells in code/ in numbered order; each writes the result files
   listed in the manifest. The temporal split is fixed at the 60th and 80th
   timestamp percentiles; graph runs use seeds {2, 3, 4} plus the original
   run, early stopping at the best-validation checkpoint (epoch ten).

## License

Code: MIT. Result files and documentation: CC-BY-4.0.

## Citation

If you use this code or these artifacts, please cite the paper (citation
updated upon acceptance) and the archived deposit:

[BibTeX placeholder: paper citation + Zenodo DOI, inserted after the v1.0
release mints the DOI]
