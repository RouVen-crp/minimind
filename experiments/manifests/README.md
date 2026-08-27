# Environment recreation

Create the Conda base environment first:

```powershell
conda create --name minimind-recreated --file conda-explicit-win-64.txt
conda activate minimind-recreated
```

Then install the pip snapshot with the PyTorch CUDA 12.4 index. This command supersedes the shorter comment at the end of `conda-explicit-win-64.txt`:

```powershell
python -m pip install -r pip-freeze.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

Verify that `torch.__version__` ends in `+cu124` and `torch.cuda.is_available()` is `True` before training.
