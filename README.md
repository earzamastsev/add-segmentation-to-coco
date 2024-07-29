# add-segmentation-to-coco
Add segmentation masks to COCO detection dataset

## Prerequests
1. Detector model put to `checkpoints` dir
- checkpoints/config.py
- checkpoints/epoch_75.pth

2. SAM weights put to `weights` dir here - [vit_h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- weights/sam_vit_h_4b8939.pth

3. COCO json files put to `dataset` dir
- dataset/annotation.json
- images dir for image files

4. Install [mmdetection](https://github.com/open-mmlab/mmdetection) and [mmyolo](https://github.com/open-mmlab/mmyolo)

5. Install modules `pip install -r requirements.txt`

6. Use help with `python segmentation_coco.py -h`
