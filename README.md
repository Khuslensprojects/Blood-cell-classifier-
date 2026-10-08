# Blood-cell-classifier-
Deep-learning classifier for 8 blood cell types from microscopy images, comparing a pixel-based baseline against a fine-tuned ResNet18. Includes error analysis and model interpretability (Grad-CAM). 
## Results (BloodMNIST, test set, 3,421 images)

| Model | Accuracy |
|---|---|
| Logistic regression (raw pixels) | 0.825 |
| Fine-tuned ResNet18 | 0.977 |

**Where it struggles:** immature granulocytes (recall 0.91, 91%) and monocytes (precision 0.87, 87%).
**Where it thrives:** platelet and eosinophil (precision, 100%) (recall, 100%)  (f1-score, 100%).
