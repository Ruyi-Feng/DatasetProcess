import os
import shutil
from PIL import Image


def convert_segmentation_to_yolo_bbox(seg_file, image_width, image_height):
    with open(seg_file, "r") as f:
        lines = f.readlines()

    bbox_lines = []
    for line in lines:
        parts = line.strip().split()
        class_id = parts[0]
        coords = list(map(float, parts[1:]))
        x_coords = coords[0::2]
        y_coords = coords[1::2]

        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        # Calculate center, width, and height
        x_center = (x_min + x_max) / 2.0 / image_width
        y_center = (y_min + y_max) / 2.0 / image_height
        bbox_width = (x_max - x_min) / image_width
        bbox_height = (y_max - y_min) / image_height

        bbox_line = f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"
        bbox_lines.append(bbox_line)

    return bbox_lines


def process_folder(root_folder):
    seg_labels_folder = os.path.join(root_folder, "seg_labels")
    yolo_labels_folder = os.path.join(root_folder, "labels")
    images_folder = os.path.join(root_folder, "images")

    # Rename the original labels folder to seg_labels
    if os.path.exists(seg_labels_folder):
        shutil.rmtree(seg_labels_folder)
    os.rename(os.path.join(root_folder, "labels"), seg_labels_folder)

    # Create the yolo labels folder
    if not os.path.exists(yolo_labels_folder):
        os.makedirs(yolo_labels_folder)

    for filename in os.listdir(seg_labels_folder):
        if filename.endswith(".txt"):
            seg_file = os.path.join(seg_labels_folder, filename)
            image_file = os.path.join(
                images_folder, filename.replace(".txt", ".jpg")
            )  # Assuming images are in .jpg format
            with Image.open(image_file) as img:
                image_width, image_height = img.size

            bbox_lines = convert_segmentation_to_yolo_bbox(
                seg_file, image_width, image_height
            )

            output_file = os.path.join(yolo_labels_folder, filename)
            with open(output_file, "w") as f:
                f.writelines(bbox_lines)
            print(f"Processed {filename}")



if __name__ == '__main__':
    root_folder = "path/to/root_folder"
    process_folder(root_folder)
