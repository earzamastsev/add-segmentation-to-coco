import matplotlib.pyplot as plt
import time
import json
from datetime import timedelta
from functools import wraps
import random
import argparse
import os


from tqdm.auto import tqdm
from PIL import Image
from pycocotools.coco import COCO

import torch
from torchmetrics.detection import IntersectionOverUnion

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import numpy as np                                 
from skimage import measure                        
from shapely.geometry import Polygon, MultiPolygon

config_file = 'checkpoints/config.py'
checkpoint_file = 'checkpoints/epoch_75.pth'

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SEGMENTATION_MODEL_TYPE = 'vit_h'

sam = sam_model_registry[SEGMENTATION_MODEL_TYPE](checkpoint='weights/sam_vit_h_4b8939.pth').to(device=DEVICE)
segment_model = SamPredictor(sam)
iot_metric = IntersectionOverUnion(box_format='xywh')

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = int(end_time - start_time)
        print(
            f"\n === {func.__name__} выполнена за {timedelta(seconds=execution_time)} ===\n")
        return result
    return wrapper


class AddSegmentationToCoco:
    def __init__(self, dataset_path, annotation_file, image_path, output_path) -> None:
        self.dataset_path = dataset_path
        self.annotation_file = os.path.join(dataset_path, annotation_file)
        self.image_path = os.path.join(dataset_path, image_path)
        self.output_path = output_path
        self.coco = COCO(self.annotation_file)
        print(self.annotation_file)
    
    def create_sub_mask_annotation(self, sub_mask):
        # Find contours (boundary lines) around each sub-mask
        # Note: there could be multiple contours if the object
        # is partially occluded
        contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

        segmentations = []
        polygons = []
        for contour in contours:
            # Flip from (row, col) representation to (x, y)
            # and subtract the padding pixel
            for i in range(len(contour)):
                row, col = contour[i]
                contour[i] = (col - 1, row - 1)

            # Make a polygon and simplify it
            poly = Polygon(contour)
            poly = poly.simplify(1.0, preserve_topology=False)
            polygons.append(poly)
            segmentation = np.array(poly.exterior.coords, dtype='int16').ravel().tolist()
            segmentations.append(segmentation)

        # Combine the polygons to calculate the bounding box and area
        multi_poly = MultiPolygon(polygons)
        x, y, max_x, max_y = multi_poly.bounds
        width = max_x - x
        height = max_y - y
        bbox = [int(i) for i in (x, y, width, height)]
        area = int(multi_poly.area)

        return segmentations, bbox, area


    def image_show(self, img_id):
        image_file_name = self.coco.loadImgs(img_id)[0]['file_name']
        image = plt.imread(os.path.join(self.image_path, image_file_name))
        annot_id = self.coco.getAnnIds(imgIds=img_id)
        annot_data = self.coco.loadAnns(annot_id)
        plt.imshow(image)
        
        # Display the specified annotations
        self.coco.showAnns(annot_data, draw_bbox=True)

        plt.axis('off')
        plt.title('Annotations for Image ID: {}'.format(img_id))
        plt.tight_layout()
        plt.show()

    def label_dataset_image(self, image_id):
        image_file_name = os.path.join(self.image_path, self.coco.loadImgs(image_id)[0]['file_name'])
        image = Image.open(image_file_name)
        segment_model.set_image(np.array(image))
        
        annot_ids = self.coco.getAnnIds(imgIds=image_id)
        for annotate in self.coco.loadAnns(annot_ids):
            bbox = [annotate['bbox'][0], annotate['bbox'][1], annotate['bbox'][0]+annotate['bbox'][2], annotate['bbox'][1]+annotate['bbox'][3]]
            bbox = np.array(bbox)
            masks, scores, _ = segment_model.predict(
                box=bbox,
                multimask_output=True,
            )
            best_mask = np.argmax(scores)
            mask = np.zeros_like(masks[best_mask, ...], dtype='uint8')
            mask[masks[best_mask, ...]==True] = 255
            segmentation, bbox_predicted, area = self.create_sub_mask_annotation(mask)
            if not segmentation:
                continue
            
            annotate['segmentation'] = segmentation
            iou = iot_metric([{'boxes':torch.tensor([annotate['bbox']]),
                            'labels':torch.tensor([0])}],
                            [{'boxes':torch.tensor([bbox_predicted]),
                            'labels':torch.tensor([0])}],
                            )
            # print(iou)
            # self.image_show(image_id)
            annotate['bbox'] = bbox_predicted
            if iou['iou'] > 0.8:
                annotate['bbox'] = bbox_predicted
                annotate['area'] = area
        

def main(dataset_path, annotation_file, image_path, output_path):
    segmentator = AddSegmentationToCoco(dataset_path, annotation_file, image_path, output_path)
    for image_id in tqdm(segmentator.coco.getImgIds()):
        try:
            segmentator.label_dataset_image(image_id)
        except Exception as e:
            print(f'Error while labelig image ID {image_id}: {e}')
    output_filename = os.path.join(output_path, annotation_file)
    os.makedirs(output_path, exist_ok=True)
    with open(output_filename, 'wt') as f:
        json.dump(segmentator.coco.dataset, f)
        print(f'Save COCO dataset with segmentation data to {output_filename}.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add segmetation masks to COCO detection dataset.")
    parser.add_argument('--dataset_path', type=str, default='dataset', help='Path to the root dataset directory.')
    parser.add_argument('--annotation_file', type=str, default='annotation.json', help='Name of the annotation file in dataset directory.')
    parser.add_argument('--image_path', type=str, default='images', help='Images dir in dataset directory.')
    parser.add_argument('--output_path', type=str, default='output', help='Directory name for dataset with segmentation.')

    args = parser.parse_args()

    main(args.dataset_path, args.annotation_file, args.image_path, args.output_path)



        
# train: 5147, val: 8482
# image_id = random.choice(coco.getImgIds())

# image_id = 6166
# image_show(image_id)

# for file in [f'{dataDir}/annotations/annotation_val.json', 
#              f'{dataDir}/annotations/annotation_train.json']: 
#     print(f'Check and coorrect segmentation data in {file}...')
#     coco=COCO(file)
#     error_count = 0
#     for annot in tqdm(coco.dataset['annotations']):
#         for idx, segment in enumerate(annot['segmentation']):
#             if segment == []:
#                 annot['segmentation'].pop(idx)
#                 error_count += 1
#     print(f'Found {error_count} empty lists in {file}')                
#     print(f'Save labeled data to {file}')
#     with open(file, 'wt') as f:
#         json.dump(coco.dataset, f)
