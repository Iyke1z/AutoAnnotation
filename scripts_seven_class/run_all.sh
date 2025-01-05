python3 -m experiments.glocal_analysis --model_name yolov8_nc7 --dataset_name coco2017 --batch_size 13

CLASSES=(0 1 2 3 4 5 6) #7
for s in "${CLASSES[@]}"
do
  python3 -m experiments.global_class_concepts --model_name yolov8_nc7 --dataset_name coco2017 --class_id $s --batch_size 4
done


python3 -m experiments.explanation_complexity --model_name yolov8_nc7 --dataset_name coco2017

LAYERS=(model.0.conv model.2.cv3.conv model.2.m.1.cv2.conv model.4.cv3.conv model.4.m.1.cv2.conv model.4.m.3.cv2.conv model.6.cv3.conv model.6.m.1.cv2.conv model.6.m.3.cv2.conv model.6.m.5.cv2.conv model.8.cv3.conv model.8.m.1.cv2.conv model.13.cv1.conv model.13.m.0.cv2.conv model.17.cv1.conv model.17.m.0.cv2.conv model.20.cv1.conv model.20.m.0.cv2.conv model.23.cv1.conv model.23.m.0.cv2.conv model.24.m.1
)
for l in "${LAYERS[@]}"
do
  python3 -m experiments.instance_perturbation_od --model_name yolov8_nc7 --dataset_name coco2017 --layer_name $l --num_samples 100 --batch_size 20 --insertion False
  python3 -m experiments.instance_perturbation_od --model_name yolov8_nc7 --dataset_name coco2017 --layer_name $l --num_samples 100 --batch_size 20 --insertion True
done


python3 -m experiments.plot_instance_perturbation --model_name yolov8_nc7 --dataset_name coco2017 --insertion False

python3 -m experiments.plot_instance_perturbation --model_name yolov8_nc7 --dataset_name coco2017 --insertion True


CLASSES=(0 1 2 3 4 5 6 )
LAYER=(model.6.cv3.conv model.8.cv3.conv model.10.conv)
for s in "${CLASSES[@]}"
do
  for l in "${LAYER[@]}"
  do
    python3 -m experiments.concept_backgroundiness --model_name yolov8_nc7 --dataset_name coco2017 --layer_name $l --class_id $s
    python3 -m experiments.concept_backgroundiness_evaluation --model_name yolov8_nc7 --dataset_name coco2017 --layer_name $l --class_id $s
  done
done

python3 -m experiments.plot_concept_backgroundiness --model_name yolov8_nc7 --dataset_name coco2017 --labels "CRP Relevance,LRP,Activation"

python3 -m experiments.plot_concept_backgroundiness --model_name yolov8_nc7 --dataset_name coco2017 --labels "CRP Relevance,Guided GradCAM,GradCAM,SSGradCAM"

