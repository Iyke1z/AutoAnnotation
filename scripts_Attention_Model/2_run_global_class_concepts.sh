
CLASSES=(0 1 2 3 4 5 6) #7
for s in "${CLASSES[@]}"
do
  python3 -m experiments.global_class_concepts --model_name yolov5_nc7_attention --dataset_name coco2017 --class_id $s --batch_size 4
  #python3 -m experiments.global_class_concepts --model_name yolov5 --dataset_name coco2017 --class_id $s --batch_size 4
done

# for rel_init in {prob,ones};do
#   echo "run ${rel_init}"
#   CLASSES=(1 2 3 4 5 6)
#   for s in "${CLASSES[@]}"
#   do
#     python3 -m experiments.global_class_concepts --model_name deeplabv3plus --dataset_name voc2012 --class_id $s --batch_size 3 --rel_init $rel_init
#   done

#   CLASSES=(0 1 2 3 4 5 6) #
#   for s in "${CLASSES[@]}"
#   do
#     python3 -m experiments.global_class_concepts --model_name unet --dataset_name cityscapes --class_id $s --batch_size 5 --rel_init $rel_init
#   done
# done