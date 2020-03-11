# Detection Tool
## Goal of this repo
if you want to use a model in detectron2 as front end to detect object in your dataset,you can use this repo to quickly try many model while don't neet to change ```config.yaml``` to  fit your dataset,only change some parameter in ```train.py```, and you can get visual result use```visual_2.py```
## How to use
Only need to change some parameter in ```train.py```
### files set
1. put config file in ```config```
2. put pretrained weight in ```pretrain```
3. ```visual_v1.py``` and evalution use same weight path
### change parameter
1. change dataset path in ```root_davis```
2. when use ```visual_v1```,recomment to set ```gpu=1```
3. if want to get else result ,such as ```daivs_test```,change ```visual_what``` in ```visual_result.py```

## visual_v1 and visual_v2
**visual_v1**:
+ launch visual_v1 in ```train.py```
+ need ```config```and```weight```

**visual_v2**:
+ independent run,change parameter
+ need result.jason
+ need register dateset
