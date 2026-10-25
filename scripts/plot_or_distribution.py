"""Plot OR distributions from an actual proposal-level evaluation CSV."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('runs/or_distribution.png'))
    args = parser.parse_args()
    frame = pd.read_csv(args.scores)
    needed = {'gt_occlusion', 'proposal_OR'}
    if not needed.issubset(frame.columns):
        parser.error('CSV must contain gt_occlusion and proposal_OR')
    samples = [frame.loc[frame.gt_occlusion == i, 'proposal_OR'].dropna().to_numpy() for i in (0,1,2)]
    if any(len(s)==0 for s in samples):
        parser.error('CSV needs observations for occlusion labels 0, 1 and 2')
    fig, ax = plt.subplots(figsize=(7.5,4.5))
    ax.boxplot(samples, labels=['Visible (0)','Partial (1)','Large (2)'])
    ax.set_ylabel('Proposal-level OR score')
    ax.set_xlabel('KITTI occlusion annotation')
    ax.set_title('OR distribution by occlusion annotation')
    fig.tight_layout()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180)
    print(args.output)

if __name__=='__main__':
    main()
