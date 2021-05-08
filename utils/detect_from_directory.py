import argparse
import time
import logging

from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

from lego_sorter_server.analysis.AnalysisService import AnalysisService
from lego_sorter_server.analysis.detection import DetectionUtils
from lego_sorter_server.analysis.detection.LegoLabeler import LegoLabeler


def is_image(file_name):
    extension = file_name.split('.')[-1]
    return extension == 'jpeg' or extension == 'jpg' or extension == 'png'


def process_images_in_path(input_path: Path, output_path: Path, analysis_service: AnalysisService, skip_images: bool,
                           save_cropped: bool, output_cropped: Path, skip_xml: bool):
    start_time = time.time()
    counter = 0

    labeler = LegoLabeler()

    for file in input_path.iterdir():
        if file.is_file() and is_image(file.name):
            xml_name = file.name.split(".")[0] + ".xml"
            dest_path_img = output_path / file.name
            dest_path_xml = output_path / xml_name
            image = Image.open(file)
            detection_results = analysis_service.detect(image, threshold=0.3, discard_border_results=True)
            width, height = image.size
            if not skip_xml:
                label_file = labeler.to_label_file(file.name, dest_path_xml, width, height, detection_results.detection_boxes)
                with open(dest_path_xml, "w") as label_xml:
                    label_xml.write(label_file)
            if not skip_images:
                image.save(dest_path_img)
            if save_cropped is True and len(detection_results.detection_boxes) > 0:
                output_cropped_images = output_cropped / file.parent.name
                output_cropped_images.mkdir(exist_ok=True, parents=False)
                for i in range(len(detection_results.detection_boxes)):
                    cropped_image_path = output_cropped_images / f"c{i}_{file.name}"
                    image_cropped = DetectionUtils.crop_with_margin(image, *detection_results.detection_boxes[i])
                    image_cropped.save(cropped_image_path)

            counter += 1

    seconds_elapsed = time.time() - start_time
    print(
        f"Processing path {input_path} took {seconds_elapsed} seconds, "
        f"{1000 * (seconds_elapsed / counter) if counter != 0 else 0} ms per image."
    )


def process_recursive(input_path: Path,
                      output_path: Path,
                      executor: ThreadPoolExecutor,
                      analysis_service: AnalysisService,
                      skip_images: bool,
                      save_cropped: bool,
                      output_cropped: Path,
                      skip_xml: bool):
    output_path.mkdir(exist_ok=True)
    dirs_to_process = []

    for file in input_path.iterdir():
        if file.is_dir():
            dirs_to_process.append(file)

    futures = []
    for directory in dirs_to_process:
        sub_out_path = (output_path / directory.name)
        futures += process_recursive(directory, sub_out_path, executor, analysis_service, skip_images, save_cropped, output_cropped, skip_xml)

    futures.append(executor.submit(process_images_in_path, input_path, output_path, analysis_service, skip_images, save_cropped, output_cropped, skip_xml))
    return futures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Detects lego bricks on images. This script copies the input directory structure.')
    parser.add_argument('-i' '--input_dir', required=True, help='A path to a directory containing images to process.',
                        type=str, dest='input')
    parser.add_argument('-o', '--output_path', required=True, help='An output path.', type=str, dest='output')
    parser.add_argument('-co', '--cropped_output_path', required=False, type=str, dest='output_cropped',
                        help='An output path for cropped bricks, if not set, \'output_path / cropped\' is being used.')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='Process images in the input_path and its subdirectories.')
    parser.add_argument('-si', '--skip_images', action='store_true', dest='skip_images',
                        help='Whether to skip copying images from the input directory to the output directory.')
    parser.add_argument('-sx', '--skip_xml', action='store_true', dest='skip_xml',
                        help='Whether to skip creating xml files with detection results. ')
    parser.add_argument('-sc', '--save_cropped', action='store_true', dest='save_cropped',
                        help='Whether to crop detected images and save them as separate files.')
    args = parser.parse_args()

    if not args.output_cropped:
        args.output_cropped = Path(args.output) / "cropped"
        args.output_cropped.mkdir(exist_ok=True, parents=True)

    logging.getLogger().disabled = True
    analysis_service = AnalysisService()

    if args.recursive:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = process_recursive(Path(args.input), Path(args.output), executor, analysis_service,
                                        args.skip_images, args.save_cropped, Path(args.output_cropped), args.skip_xml)
            for future in futures:
                future.result()
    else:
        process_images_in_path(Path(args.input), Path(args.output), analysis_service, args.skip_images,
                               args.save_cropped, Path(args.output_cropped), args.skip_xml)
