# MNIST MLP — Results Summary

- Checkpoint: `outputs/checkpoints/best_model.pt` (epoch 36)
- Test accuracy: **0.9865**
- Test macro F1: **0.9864**
- Acceptance criteria (accuracy >= 0.98 OR macro F1 > 0.90): **PASS**

## Most confused digit pairs
- true=9 predicted=4: 10 cases
- true=4 predicted=9: 8 cases
- true=5 predicted=3: 6 cases
- true=4 predicted=6: 5 cases
- true=9 predicted=7: 4 cases

## Artifacts
- `outputs/reports/confusion_matrix.png`
- `outputs/reports/classification_report.txt`
- `outputs/logs/history.csv`
