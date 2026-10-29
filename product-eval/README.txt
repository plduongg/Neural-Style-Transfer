CAPE model:
- checkpoints/cape/decoder_iter_30000.pth
- checkpoints/cape/embedding_iter_30000.pth
- checkpoints/cape/transformer_iter_30000.pth
- checkpoints/cape/checkpoint_iter_30000.pth

RPA model:
- checkpoints/rpa/decoder_iter_30000.pth
- checkpoints/rpa/embedding_iter_30000.pth
- checkpoints/rpa/transformer_iter_30000.pth
- checkpoints/rpa/checkpoint_iter_30000.pth

Main inference file:
- code/test.py

Typical test command:
- python test.py --content_dir data/content --style_dir data/style --output output

Test image size:
- 512

Preprocessing:
- same preprocessing for both models

RPA modification:
- modified transformer in code/models/transformer_rpa.py

Notes:
- both models were trained on the same dataset
- both models are compared at iteration 30000