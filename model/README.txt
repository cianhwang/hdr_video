HDResNet

Requirements: saved in file environment.yaml
        to install the conda environment: conda env create --file environment.yaml
Training: python train.py --merge_ver [m/mbp] --note "Remarks"
Training(continue): python train.py --merge_ver [m/mbp] --note "Remarks" --ckpt_dir ckpt/[xxx] --logs_dir runs/[xxx] --restore_file [best/last]
