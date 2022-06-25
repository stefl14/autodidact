import json
from typing import Union

import click
import layoutparser as lp
import numpy as np
from layoutparser.elements.layout_elements import TextBlock
from pathlib import Path


def perform_ocr(
    ocr_agent: Union[lp.TesseractAgent, lp.GCVAgent],
    image: np.array,
    block: TextBlock,
    page_num: int,
    left_pad: int = 15,
    right_pad: int = 5,
    top_pad: int = 5,
    bottom_pad: int = 5,
) -> None:
    """
    Perform OCR on a block of text.

    Args:
        ocr_agent: The OCR agent to use.
        image: The crop to perform OCR on.
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
    # TODO: Clean this up with custom data class.
    block.__setattr__(
        "page_num", page_num
    )  # Needed for reading order detection downstream. Bit messy.


@click.command()
@click.option(
    "-i",
    "--input-dir",
    type=click.Path(exists=True),
    required=True,
    default="../downloads",
    help="The directory to read PDFs from.",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(exists=True),
    required=True,
    default="../data/ocr",
)
@click.option(
    "--ocr-agent", type=click.Choice(["tesseract", "gcv"]), required=True, default="gcv"
)
@click.option(
    "-m",
    "--model",
    type=str,
    required=True,
    help="The model to use for OCR.",
    default="mask_rcnn_X_101_32x8d_FPN_3x",  # powerful detectron2 model.
)
@click.option("-t", "--detectron-threshold", type=float, default=0.5)
def run_cli(
    input_dir: Path,
    output_dir: Path,
    ocr_agent: str,
    model: str,
    detectron_threshold: float = 0.5,
) -> None:
    """
    Run cli to extract semi-structured JSON from document-AI + OCR.

    Args:
        input_dir: The directory containing the PDFs to parse.
        output_dir: The directory to write the parsed PDFs to.
        ocr_agent: The OCR agent to use.
        model: The document AI model to use.
        detectron_threshold: The threshold to use for Detectron2.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    model = lp.Detectron2LayoutModel(
        config_path=f"lp://PubLayNet/{model}",  # In model catalog,
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        extra_config=[
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
            detectron_threshold,
        ],  # Optional
    )

    if ocr_agent == "tesseract":
        ocr_agent = lp.TesseractAgent(languages="eng")
    elif ocr_agent == "gcv":
        ocr_agent = lp.GCVAgent(languages="eng")
    input_dir = Path(input_dir)
    for file in input_dir.iterdir():
        file_name = file.name
        if not file_name.endswith(".pdf"):
            continue
        pdf_tokens, pdf_images = lp.load_pdf(file, load_images=True)
        for ix, image in enumerate(pdf_images):
            image_array = np.array(image)
            layout = model.detect(image_array)  # perform computer vision
            # perform ocr on extracted blocks.
            text_blocks = lp.Layout([b for b in layout if b.type == "Text"])
            # convert to CustomTextBlock to add page_num attribute.
            for block in text_blocks:
                perform_ocr(
                    ocr_agent, image_array, block, page_num=ix + 1
                )  # modify text blocks in-place

        # save extracted layout as json
        text_block_dict = text_blocks.to_dict()
        file_name_without_ext = file_name.split(".")[0]
        with open(output_dir / f"{file_name_without_ext}.json", "w") as f:
            json.dump(text_block_dict, f)


if __name__ == "__main__":
    run_cli()
