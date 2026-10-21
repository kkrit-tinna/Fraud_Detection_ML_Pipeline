"""Generate reports/results.ipynb from the files written by `make pipeline`.

    python scripts/build_results_notebook.py && jupyter nbconvert --execute --inplace reports/results.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
    md("# Credit card fraud detection: results\n\n"
       "This notebook reads the reports written by `make pipeline` (it doesn't need the raw data). "
       "All numbers are on the untouched `fraudTest.csv` (555,719 transactions, Jun–Dec 2020). "
       "Decision thresholds were tuned on a time-based validation split of `fraudTrain.csv`.\n\n"
       "Error-rate definitions: **Type II error rate** = missed frauds / all frauds (= 1 − recall); "
       "**missed frauds / all transactions** = FN / N; **Type I error rate** = false alarms / legitimate transactions."),
    code("import json\nimport pandas as pd\nfrom IPython.display import Image, display\n\n"
         "pd.set_option('display.float_format', '{:.4f}'.format)\n"
         "comparison = pd.read_csv('model_comparison.csv')\ncomparison"),
    md("## Selected model"),
    code("report = json.load(open('model_comparison.json'))\n"
         "best = report['selected_model']\nr = report['results'][best]\n"
         "print('Selected:', best)\nprint('Hyperparameters:', r['best_params'])\n"
         "pd.DataFrame({'validation': r['validation'], 'test': r['test']})"),
    md("## Hyperparameter search (validation PR-AUC)"),
    code("pd.DataFrame([{'model': m, 'params': t['params'], 'val_pr_auc': t['val_pr_auc'], 'seconds': t['seconds']}\n"
         "              for m, res in report['results'].items() for t in res['tuning_trials']])"),
    md("## ROC and precision-recall curves"),
    code("display(Image('figures/roc_pr_curves.png'))"),
    md("## Confusion matrix and feature importance"),
    code("display(Image('figures/confusion_matrix.png'))\ndisplay(Image('figures/feature_importance.png'))"),
    md("## Monitoring: month-by-month replay of the test period\n\n"
       "Alerts fire when recall drops more than 5 points below validation or any feature's PSI exceeds 0.2."),
    code("monthly = pd.read_csv('monitoring/monthly_performance.csv')\ndisplay(monthly)\n"
         "display(Image('monitoring/monthly_performance.png'))"),
]
out = ROOT / "reports" / "results.ipynb"
nbf.write(nb, out)
print(f"wrote {out}")
