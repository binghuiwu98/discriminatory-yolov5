#!/bin/bash
python voc_label.py
mkdir Train
mkdir Val
mkdir Test
python 2COCO.py --image_path ./train.txt --save ./train.json
python 2COCO.py --image_path ./val.txt --save ./val.json
python 2COCO.py --image_path ./test.txt --save ./test.json
