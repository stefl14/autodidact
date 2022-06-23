from copy import copy
from typing import Union
import click
from functools import partial

import layoutparser as lp
from layoutparser.elements.layout_elements import TextBlock
from pathlib import Path


def perform_ocr(
    ocr_agent: Union[lp.TesseractAgent, lp.GCVAgent],
    image: lp.Image,
    block: TextBlock,
    left_pad: int = 15,
    right_pad: int = 5,
    top_pad: int = 5,
    bottom_pad: int = 5,
) -> None:
    """
    Perform OCR on a block of text.

    Args:
        image: The crop to perform OCR on.
        ocr_agent: The OCR agent to use.
        block: The block to set the text of.
        left_pad: The number of pixels to pad the left side of the block.
        right_pad: The number of pixels to pad the right side of the block.
        top_pad: The number of pixels to pad the top of the block.
        bottom_pad: The number of pixels to pad the bottom of the block.
    """
    # Pad to improve OCR accuracy as it's fairly tight.
    segment_image = block.pad(
        left=left_pad, right=right_pad, top=top_pad, bottom=bottom_pad
    ).crop_image(image)

    # Perform OCR
    text = ocr_agent.detect(segment_image, return_only_text=True)

    # Save OCR result
    block.set(text=text, inplace=True)


@click.command()
@click.option("--input-dir", type=click.Path(exists=True), required=True)
@click.option("--output-dir", type=click.Path(exists=True), required=True)
@click.option("--ocr-agent", type=click.Choice(["tesseract", "gcv"]), required=True)
@click.option("--model", type=str, required=True)
def run_cli(input_dir: Path, output_dir, model) -> None:
    """
    Run the script from the command line.
    """
    if model == "tesseract":
        ocr_agent = lp.TesseractAgent(model_path=model)
    elif model == "gcv":
        ocr_agent = lp.GCVAgent(model_path=model)
    input_dir = Path(input_dir)
    for file in input_dir.iterdir():
        file_name = file.name
        pdf_tokens, pdf_images = lp.load_pdf(file, load_images=True)
        for image in pdf_images:
            layout = model.detect(image) # perform computer vision
            # perform ocr on extracted blocks.
            text_blocks = lp.Layout([b for b in layout if b.type == "Text"])
            ocr_func = partial(perform_ocr, image=image)
            for block in text_blocks:
                ocr_func(block) # modify text blocks in-place
        # save extracted layout as json
        text_blocks.astype(dict).to_json(output_dir / f"{file_name}.json")





if __name__ == '__main__':
    google_ocr_agent = lp.GCVAgent(languages="eng")
    run_cli(ocr_agent=google_ocr_agent)