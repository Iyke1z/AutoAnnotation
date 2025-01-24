
python3 -m experiments.plot_concept_backgroundiness --model_name yolov5_nc7_attention --dataset_name coco2017 --labels "CRP Relevance,LRP,Activation"

python3 -m experiments.plot_concept_backgroundiness --model_name yolov5_nc7_attention --dataset_name coco2017 --labels "CRP Relevance,Guided GradCAM,GradCAM,SSGradCAM"
