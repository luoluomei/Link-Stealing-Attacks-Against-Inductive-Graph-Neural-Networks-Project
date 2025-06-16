## Project Structure

```
Link-Stealing-Attacks-.../
|
├── link_steal_pets2024/
│   ├── train_gnn.py             # Target/Shadow GNN training script
│   ├── mlp_attack.py            # MLP-based attack implementation
│   ├── run_attack.py            # Customizable experiment runner
│   └── GraphGallery/            # Graph utility package (external dependency)
|
├── output/                      # Logs and experiment results
│   ├── logs/
│   └── results/
```

## Setup Instructions

Please run the following in Google Colab or a CUDA-enabled local machine.

```bash
# Clone project and dependencies
!git clone https://github.com/luoluomei/Link-Stealing-Attacks-Against-Inductive-Graph-Neural-Networks-Project.git
%cd /content/Link-Stealing-Attacks-Against-Inductive-Graph-Neural-Networks-Project/link_steal_pets2024

!git clone https://github.com/luoluomei/GraphGallery.git
%cd GraphGallery
!pip install -e . --verbose
%cd ..

# Install necessary packages
!pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
!pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install torchdata==0.6.1
!pip install networkx matplotlib seaborn pandas scikit-learn tqdm yacs tabulate gensim

# Fix compatibility bug in GraphGallery tqdm
!sed -i 's/tqdm.__doc__ = tqdm_base.__doc__ + tqdm_base.__init__.__doc__/tqdm.__doc__ = (tqdm_base.__doc__ or "") + (tqdm_base.__init__.__doc__ or "")/' /content/link_steal_pets2024/GraphGallery/graphgallery/utils/tqdm.py
```

## Running the Full Experiment Suite

The script `run_attack()` can be called from a Python environment. It will:

1. Train the target model (if not already present)
2. Train shadow models for each dataset-model combination
3. Run selected attack methods using `mlp_attack.py`
4. Save AUC scores into `./output/results/attack{ID}_summary.csv`

### Example Call:

```python
from run_attack import run_attack

run_attack(
    target_dataset="cora_ml",
    shadow_datasets=["cora_ml", "dblp"],
    shadow_models=["graphsage", "gin"],
    attack_ids=[0, 1, 2],
    props=[100],
    seed_num=5,
    gpu=0
)
```

## Attack Method Index (attack_id)

| ID | Method                        | Description                          |
|----|-------------------------------|--------------------------------------|
| 0  | 0-hop_posteriors              | Basic topology-free MIA              |
| 1  | 1-hop_posteriors              | Include 1-hop node information       |
| 2  | 2-hop_posteriors              | Include 2-hop node information       |
| 3  | 0-hop_posteriors_node         | Node-only view, 0-hop                |
| 4  | 1-hop_posteriors_node         | Node-only view, 1-hop                |
| 5  | 2-hop_posteriors_node         | Node-only view, 2-hop                |
| 6  | 1-hop_posteriors_graph        | Graph-structure-enhanced MIA         |
| 7  | 2-hop_posteriors_graph        | Graph-structure-enhanced MIA         |
| 8  | 1-hop_posteriors_node_graph   | Node + Graph hybrid attack (1-hop)   |
| 9  | 2-hop_posteriors_node_graph   | Node + Graph hybrid attack (2-hop)   |

## Output Format

Each experiment's result is saved in `./output/results/attack{ID}_summary.csv` with the following schema:

| target_dataset | shadow_dataset | shadow_model | attack_id | prop | seed | test_auc |
|----------------|----------------|--------------|-----------|------|------|----------|

Example:

```
cora_ml,dblp,graphsage,0,100,1234,0.8012
cora_ml,dblp,graphsage,0,100,4321,0.7935
...
cora_ml,dblp,graphsage,0,100,avg,0.7974
```

## Notes

- Make sure to delete or rename old shadow model checkpoints to avoid silent reuse.
- All models are trained with fixed hyperparameters as used in the original paper.
- The script will automatically skip training if the model file already exists.

## Contact

For questions or contributions, please contact:

**Iris Hong**  
h.yihan@wustl.edu


