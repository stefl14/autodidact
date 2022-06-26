poetry env remove autodidact-n6tHuUMF-py3.8
poetry shell
poetry install
poe force-cuda11 = "pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113"
# Detectron 2 model for document AI.
poe layout-parser-base = "pip3 install layoutparser"
poe layout-parser-models = "pip3 install 'layoutparser[layoutmodels]'"
poe layout-parser-ocr = "pip3 install 'layoutparser[ocr]'"
poe detectron-2 = "pip3 install 'git+https://github.com/facebookresearch/detectron2.git@v0.5'"
