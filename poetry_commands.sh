pip install --upgrade pip
poetry env remove autodidact-n6tHuUMF-py3.8
poetry shell
poetry install
poe force-cuda11
# Detectron 2 model for document AI.
poe layout-parser-base
poe layout-parser-models
poe layout-parser-ocr
poe detectron-2
