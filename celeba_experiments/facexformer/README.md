# FaceXFormer network code

`model.py` and `transformer.py` are copied from the official FaceXFormer repository, https://github.com/Kartik-3004/facexformer (`network/models/facexformer.py` and `network/models/transformer.py`), under its MIT License (`LICENSE` in this directory).

Changes from the original:

- `model.py` imports the transformer module by its package path.
- `model.py` builds the Swin-B backbone without ImageNet weights (`swin_b(weights=None)`), because the released FaceXFormer checkpoint replaces all of them. This avoids an extra download and does not change the loaded model.

`../evaluate.py` loads the released checkpoint (`kartiknarayan/facexformer`, `ckpts/model.pt`) from the Hugging Face hub.

