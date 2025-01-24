import os
from glob import glob
import cv2
root_dir_imgs = "/home/projects/xai/exaplaiable_ai/ALM/L-CRP/datasets/coco_yolo/images"
root_dir = "/home/projects/xai/exaplaiable_ai/ALM/L-CRP/datasets/coco_yolo/labels"
train_dir = os.path.join(root_dir, "train_ori")
val_dir = os.path.join(root_dir, "val_ori")

def noramlize_labels(label_dir):
    for path in glob(os.path.join(label_dir, "**")):
        label_name = path.split("/")[-1]
        image_name = label_name[:-4]+".jpg"
        type_op = os.path.basename(label_dir).split("_")[0]
        new_path = os.path.join(os.path.dirname(label_dir),type_op, label_name.split("_")[0])
        image_path = os.path.join( root_dir_imgs,type_op, image_name)
        img = cv2.imread(image_path)
        img_h, img_w, _ = img.shape
        with open(path, "r") as fp:
            with open(new_path, "w") as fp_out:
                for line in fp.readlines():
                    class_id, x1,y1,w,h = line.split(" ")
                    x1 = float(x1)
                    y1 = float(y1)
                    w = float(w)
                    h = float(h)
                    center_x = (x1+w/2)/(img_w)
                    center_y = (y1+h/2)/(img_h)
                    norm_w = w/img_w
                    norm_h =h/img_h

                    new_line = "{} {} {} {} {}\n".format(class_id, center_x, center_y, norm_w, norm_h )
                    fp_out.write(new_line)
                    new_x = int(center_x * img_w - w / 2)
                    new_y = int(center_y * img_h - h / 2)
                    cv2.rectangle(img, (new_x, new_y), (new_x + int(norm_w * img_w), new_y + int(norm_h * img_h)), (0, 255, 0), 2)
                    print(new_line)



if __name__ == '__main__':
    noramlize_labels(train_dir)
    noramlize_labels(val_dir)




