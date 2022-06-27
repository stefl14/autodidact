import json
from typing import Union, List

import click
import layoutparser as lp
import numpy as np
import loguru
from layoutparser.elements.layout_elements import TextBlock
from pathlib import Path
from tqdm import tqdm


def perform_ocr(
    ocr_agent: Union[lp.TesseractAgent, lp.GCVAgent],
    image: np.array,
    block: TextBlock,
    left_pad: int = 15,
    right_pad: int = 5,
    top_pad: int = 5,
    bottom_pad: int = 5,
) -> None:
    """Perform OCR on a block of text.

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


def get_text_blocks(image: np.array, model) -> lp.Layout:
    """Get the text blocks from an image using layoutparser Document-AI.

    Args:
        image: np.array of the image to get the text blocks from.
        model: The computer vision model to use for text detection.

    Returns:
        A layoutparser Layout object of text blocks.
    """
    image_array = np.array(image)
    layout = model.detect(image_array)  # perform computer vision
    # perform ocr on extracted blocks.
    text_blocks = lp.Layout([b for b in layout if b.type == "Text"])
    return text_blocks


def disambiguate_overlapping_blocks():
    pass

def detect_num_cols(text_blocks: lp.Layout) -> int:
    """Detect the number of text columns on a pdf page for processing reading order downstream.
    """
    pass

def detect_reading_order(text_blocks: lp.Layout) -> lp.Layout:
    """Detect the reading order of text blocks on a pdf page.
    """
    pass


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
    loguru.logger.info(f"Using {ocr_agent} OCR agent.")
    loguru.logger.info(f"Using {model} model.")
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
    loguru.logger.info(f"Iterating through files.")
    input_dir = Path(input_dir)
    for file in tqdm(input_dir.iterdir(), desc="Files"):
        if file.suffix != ".pdf":
            continue
        _, pdf_images = lp.load_pdf(file, load_images=True)
        pages=[]
        for ix, image in tqdm(
            enumerate(pdf_images), total=len(pdf_images), desc=file.name
        ):
            image_array = np.array(image)
            text_blocks = get_text_blocks(image_array, model)
            for block in text_blocks:
                perform_ocr(
                    ocr_agent, image_array, block
                )  # modify text blocks in-place

            # save extracted layout as json
            text_block_dict = text_blocks.to_dict()
            for dic in text_block_dict["blocks"]:
                dic["page_num"] = ix + 1

            pages.append(text_block_dict)


        out_dict = {"pages": pages}
        # # Post-processing.
        # text_block_dict = postprocess_ocr_results(text_blocks, block_pages)
        file_name_without_ext = file.name.split(".")[0]
        with open(output_dir / f"{file_name_without_ext}.json", "w") as f:
            json.dump(out_dict, f)
        loguru.logger.info(f"Saved {file_name_without_ext}.json to {output_dir}.")


if __name__ == "__main__":
    run_cli()
