python crop.py
# crop objects from iamges according to labels

python split_train_val.py
# split dataset into train.txt trainval.txt, val.txt test.val

python voclabel.py
# convert voc labels to yolo labels

python copytxt.py
# copy images to Train, Val, Test according to train, val, test .txt

python 2COCO.py --image_path ./train.txt --save ./train.json
# convert yolo labels to coco labels .json file

# need run 'python  split_train_val.py' first
bash 2coco.sh
# Create train, test, val coco json files.



