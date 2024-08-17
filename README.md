# add-segmentation-to-coco
Add segmentation masks to COCO detection dataset

## Prerequests

1. SAM weights put to `weights` dir here - [vit_h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- weights/sam_vit_h_4b8939.pth

2. COCO json files put to `dataset` dir
- dataset/annotation.json
- dataset/images dir for image files

3. Install modules `pip install -r requirements.txt`

4. Use help with `python segmentation_coco.py -h`
