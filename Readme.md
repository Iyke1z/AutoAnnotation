# This Automated Annotation Pipeline Repo is still under buildup.

# How-To Run:

1. install requirements (tested with Python 3.8.10)
2. setup dataset and models
   1. datasets: download via public available links and setup paths in corresponding dataset python files in /datasets.
      a. OpenImages: https://storage.googleapis.com/openimages/web/index.html
      b. coco2017: https://cocodataset.org

   2. models: download yolov8 checkpoints and setup paths in corresponding model python files in /models.
      2. yolov8m: https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md (model python file is adapted from there)
	
	# create own venv and activate using source .....venv/bin/activate

	# To trian own own yolov8m model with 7 class
	python train.py --data coco_seven_class.yaml --epochs 1 --weights yolov8m.pt  --batch-size 4

	# save state_dict from 7 class trained yolov8m model after train completes
	python export_weight_sd_seven_class_coco.py 

3. run glocal analysis through /scripts/run_glocal_analysis.sh

4. gather concept attributions for all detections via /scripts/run_global_class_concept.sh

5. explanation_complexity experiments
   1. run /scripts/run_plot_explanation_complexity.sh

6. faithfulness experiments
   1. run /scripts/run_instance_perturbation.sh
   2. run /scripts/run_plot_instance_perturbation.sh

7. concept_context experiments
   1. run /scripts/run_concept_backgroundiness.sh 
   2. run /scripts/run_plot_concept_backgroundiness.sh

8. CRP explanations can be run via /experiments/glocal_analysis.py

9. Most relevant reference samples (masked) can be computed via /experiments/get_reference_samples.py
