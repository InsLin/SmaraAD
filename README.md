# SamaraAD

The Offical Code of CVPR2024: Adversarial Distillation Based on Slack Matching and Attribution Region Alignment

This repo is under construction.


# Usage
## Slack Matching

You can implement $L_{sm}$ by doing the following:
```
from spearman_soft_rank import spearman_soft_rank

y_student = student_model(x)
y_teacher = teacher_model(x)
loss = spearman_soft_rank(y_student, y_teacher, beta)
```

## Attribution Region Alignment

You can easily get the class activation mapping for the model using [this repo](https://github.com/jacobgil/pytorch-grad-cam) and then use MSE to compute $L_{align}$.
