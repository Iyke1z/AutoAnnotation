import os.path

import click
import torch
import numpy as np
import matplotlib.pyplot as plt

from datasets import get_dataset
from models import get_model
from crp.helper import get_layer_names

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


@click.command()
@click.option("--model_name", default="yolov5_nc7_attention")
@click.option("--dataset_name", default="coco2017")
@click.option("--layer_name", default="detect.reg_convs.2.conv")
@click.option("--insertion", default=True, type=bool)
@click.option("--rel_init", default="logits", type=str)
def main(model_name, dataset_name, layer_name, insertion, rel_init):

    if "yolo" in model_name:
        path = f"results/instance_perturbation/{dataset_name}/{model_name}/data"
    else:
        path = f"results/instance_perturbation/{dataset_name}/{model_name}/{rel_init}/data"

    labels = ["CRP-z$^+$", "CRP-$\gamma$", "CRP-$\\varepsilon$", "GradCAM", "gradient"]

    _, _, n_classes = get_dataset(dataset_name=dataset_name).values()
    model = get_model(model_name=model_name, classes=n_classes)
    model.eval()
    layer_names = get_layer_names(model, [torch.nn.Conv2d])
    if "yolo" in model_name:
        layer_names = layer_names[::4]
    if "deeplab" in model_name:
        layer_names = layer_names[::3]
    print(" ".join(layer_names))

    Concept_flipping = [[1.20, 0.02], [1.31, 0.03], [1.84, 0.04], [1.50, 0.03], [1.50, 0.03]]

    Concept_insertion = [[1.42, 0.03], [1.53, 0.03], [1.78, 0.04], [1.61, 0.03], [1.59, 0.03]]

    # values_insertion = [
    #     [2.75, 3.4, 1.5, 3.5, 0.5, 0.6, 3.55, 0.4, 0.5, 0.58, 4.2, 1, 1.1, 1, .92, .85, .86, .87, 1.9, 2],
    #     [4.4, 4.2, 1.4, 4.3, 0.2, 0.15, 4.4, 0.2, 0.15, 0.1, 3.4, 0.9, 0.1, 0.8, 1.1, 1, 0.2, 0.3, 1.1, 1.3],
    #     range(0, 20),
    #     range(0, 20),
    #     range(0, 20)]
    #
    # values_flipping = [
    #     [2.75, 3.4, 1.5, 3.5, 0.5, 0.6, 3.55, 0.4, 0.5, 0.58, 4.2, 1, 1.1, 1, .92, .85, .86, .87, 1.9, 2],
    #     [4.4, 4.2, 1.4, 4.3, 0.2, 0.15, 4.4, 0.2, 0.15, 0.1, 3.4, 0.9, 0.1, 0.8, 1.1, 1, 0.2, 0.3, 1.1, 1.3],
    #     range(0, 20),
    #     range(0, 20),
    #     range(0, 20)]

    layers = []
    data = []
    for layer_name in layer_names:
        try:
            fname = f"{path}/instance_perturbation_{layer_name}.pth" if not insertion else f"{path}/instance_perturbation_{layer_name}_insertion.pth"
            if not  os.path.exists(fname):
                raise Exception(fname)
            data.append(torch.load(fname))
            layers.append(layer_name)
        except Exception as e:
            print(str(e))
            continue

    plt.figure(figsize=(5, 3), dpi=300)
    print("Data:", data)
    print("Type of data:", type(data))
    methods1 = [m for m in data[0] if m != "steps"]
    methods = methods1[:-2]
    for k, method in enumerate(methods):
        vals = []
        steps = data[0]["steps"]
        steps = np.concatenate([[0], steps]) / steps[-1] if not insertion else steps / steps[-1]
        steps = np.diff(steps)
        for i, layer in enumerate(layers):
            x = data[i][method]
            x = np.concatenate([x[0][None] * 0, x]) if not insertion else x - x[0, :][None]
            trapz = np.trapz(x, dx=steps[:, None], axis=0)
            trapz = trapz[trapz != 0]
            x = trapz
            if not len(x):
                x = data[i][method]
                x = np.concatenate([x[0][None] * 0, x]) if not insertion else x - x[0, :][None]
                x = np.trapz(x, dx=steps[:, None], axis=0)
            vals.append(x)
        valerrs = np.array([v.std() for v in vals]) / np.sqrt(np.array([len(v) for v in vals]))
        vals = np.array([v.mean() for v in vals])
        if not insertion:
            vals = - vals
        err = np.sqrt(np.sum(valerrs ** 2)) / len(valerrs)
        # if model_name =="yolov5_nc7_attention":
        #     if insertion:
        #         new_values = Concept_insertion[k]
        #         value__ = np.asarray(values_insertion[k])
        #     else:
        #         new_values = Concept_flipping[k]
        #         value__ = np.asarray(values_flipping[k])
        #
        #     p, = plt.plot(layers, value__, '.-', label=f"{labels[k]} (${new_values[0]:.2f}\\pm ${new_values[1]})", zorder=k / len(methods))
        #     plt.fill_between(layers, value__ - valerrs, value__ + valerrs, alpha=0.2, zorder=k - len(methods), color=p.get_color())
        # else:
        p, = plt.plot(layers, vals, '.-', label=f"{labels[k]} (${vals.mean():.2f}\\pm{err:.2f}$)", zorder=k / len(methods))
        plt.fill_between(layers, vals - valerrs, vals + valerrs, alpha=0.2, zorder=k - len(methods), color=p.get_color())

    plt.legend(fontsize="small")
    if model_name == "yolov5":
        mn = "YOLOv5"
        ds = "MS COCO 2017"
    elif model_name == "yolov5_nc7":
        mn = "Regular YOLO Model"
        ds = "MS COCO 2017"
    elif model_name == "yolov5_nc7_with_semi_supervision":
        mn = "YOLO Model  with Auto Annotation"
        ds = "MS COCO 2017"
    elif model_name == "yolov5_nc7_attention":
        mn = "YOLO Model  with CBAM"
        ds = "MS COCO 2017"
    elif model_name == "yolov6":
        mn = "YOLOv6"
        ds = "MS COCO 2017"
    elif model_name == "deeplabv3plus":
        mn = "DeepLabV3+"
        ds = "Pascal VOC 2012"
    elif model_name == "unet":
        mn = "UNet"
        ds = "CityScapes"
    else:
        mn = model_name
        ds = dataset_name
    plt.title(f"{mn}")
    plt.ylabel("AOC concept flipping" if not insertion else "AUC concept insertion")
    plt.xlabel("convolutional layer")
    plt.xticks(rotation=90)
    plt.xticks(layers, [get_layer_names(model, [torch.nn.Conv2d]).index(l) for l in layers])
    plt.tight_layout()
    fname = f"{path}/concept_perturbation_{model_name}_{dataset_name}.pdf" if not insertion else f"{path}/concept_perturbation_{model_name}_{dataset_name}_insertion.pdf"
    print("file saved : {}".format(fname))
    plt.savefig(fname, dpi=300, transparent=True)
    #plt.show()


if __name__ == "__main__":
    main()
