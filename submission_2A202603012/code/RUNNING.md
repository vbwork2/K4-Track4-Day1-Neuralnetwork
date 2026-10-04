# Run the completed lab

The completed `lab.ipynb` contains its own implementation bundle. You do not need to push changes to GitHub or upload individual Python files.

1. Open Google Colab and upload this `lab.ipynb`.
2. Select Runtime > Change runtime type > T4 GPU (or an available CUDA GPU).
3. Student ID is already `2A202603012`. Run all cells and approve the Drive mount.
4. The notebook clones the official data repository, installs the completed modules in `MyDrive/NeuralNetworkLab/submission_2A202603012/code/`, and runs Parts 0-4. Changed existing modules are backed up outside the submission first. Set `INSTALL_BUNDLED_CODE = False` if you later edit persistent modules yourself.
5. The plan uses 20 epochs for every experiment: three SGD learning rates, two extra baseline seeds, two Adam learning rates, MSE, batch size, dropout, normal/high-rate clipping, FP16, supported BF16, Xavier, and zeros. The best validation checkpoint is selected within each run. Final configuration ranking uses seed 1 validation macro-F1.
6. A submitted run, an incompatible environment, or logs without checkpoints automatically start a fresh run under `.lab_reruns/run-<unique-id>/submission_2A202603012/`. Existing submitted evidence is preserved. Best weights stay in `.lab_checkpoints/`, outside that run's submission folder. An unfinished compatible run can reuse completed experiments when all checkpoints are present. Use the printed output path for the new results; do not retune the submitted configuration after final eval.
7. Review `REPORT.md` and its measured interpretations. The report is a generated draft, not a substitute for your own explanation. Add your name. Download/open `experiments.xlsx` in Excel and save to recalculate template formulas, then replace the Drive copy.
8. After all cells complete, use File > Download > Download .ipynb in Colab. The editor's executed notebook is different from the file Python installed in Drive. Run the optional packaging snippet below in a new cell, upload the downloaded executed notebook, and download the resulting ZIP. Packaging rejects an unexecuted notebook or notebook errors.

```python
from google.colab import files
from submission import package_submission
import json
uploaded = files.upload()
name = next(name for name in uploaded if name.endswith('.ipynb'))
notebook = json.loads(uploaded[name])
code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
if any(cell.get('execution_count') is None for cell in code_cells):
    raise ValueError('Upload the executed lab notebook, with outputs.')
if any(output.get('output_type') == 'error' for cell in code_cells for output in cell.get('outputs', [])):
    raise ValueError('The notebook contains an error output.')
(OUT_DIR / 'code/lab.ipynb').write_bytes(uploaded[name])
archive = package_submission(OUT_DIR)
files.download(str(archive))
```

If `REPORT.generated.md` appears, your existing edited report was preserved. Review and merge the measured draft into `REPORT.md` before packaging.

For a local validation run from the repository root:

```powershell
python -m pytest code/test_lab.py -q
python code/run_lab.py --smoke --device cpu
```

Smoke outputs stay under `.lab_validation/` and must not be submitted. To train locally instead, use `python code/run_lab.py --student-id 2A202603012`. The local installed PyTorch currently has no CUDA support; the Colab notebook uses its preinstalled CUDA build. No local CUDA installation is needed for Colab.

After changing implementation modules, regenerate the notebook bundle with `python code/build_notebook.py`. All code comments are in English.
