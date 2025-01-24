# The Automated Annotation Pipeline repository is currently under development and will be updated progressively to include additional features and refinements.

## Create your own VirtualEnv environemnt using:
python3 -m venv myenv

## Activate VirtualEnv by
source myenv/bin/activate

## Install requirements 
pip install -r requirements.txt

## To keep consistency, Current directory to run bash and python files are project root not the script it is running from

## How-To Run:
   1. Datasets: download via public available links and setup paths in corresponding dataset python files in /datasets.
      1. OpenImagesV7: (https://storage.googleapis.com/openimages/web/index.html)
      2. COCO2017: https://cocodataset.org
      
   2. Models Training: 
      * To train Regular Model : bash yolov8_old/train_coco_Regular_Model.sh
      * To train Attention Model : bash yolov8_old/train_coco_Attention_Model.sh
      * To train Regular Model without Auto Annotation : bash yolov8_old/train_coco_Regular_Model_Without_Auto_Annotation.sh
      * To train Regular Model with Auto Annotation : bash yolov8_old/train_coco_Regular_Model_With_Auto_Annotation.sh
      
   3. Export Trained models to XAI framework:
      * To export Regular Model weight : python3 yolov8_old/export_weight_sd_seven_class_coco.py
      * To export Attention Model weight : python3 yolov8_old/export_weight_sd_seven_class_coco_attention.py
      * To export Regular Model weight without Auto Annotation : python3 yolov8_old/export_weight_sd_seven_class_coco_without_auto_annotation.py
      * To export Regular Model weight with Auto Annotation : python3 yolov8_old/export_weight_sd_seven_class_coco_with_auto_annotation.py
      
   4. Analysis Scripts location (script_location) are following. each directory has their own scripts and running them will give analysis results.:    
      * To analyse Regular Model : scripts_Regular_Model/**.sh
      * To analyse model with CBAM attention : scripts_Attention_Model/**.sh
      * To analyse model with auto annotation : scripts_Auto_Annotation/**.sh

** Download datasets.zip and unzip it too AAnno_Final/experiments/ here.
   
