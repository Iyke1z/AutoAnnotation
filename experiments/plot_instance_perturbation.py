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
@click.option("--model_name", default="yolov5")
@click.option("--dataset_name", default="coco2017")
@click.option("--layer_name", default="detect.reg_convs.2.conv")
@click.option("--insertion", default=True, type=bool)
@click.option("--rel_init", default="logits", type=str)
def main(model_name, dataset_name, layer_name, insertion, rel_init):

    if "yolo" in model_name:
        path = f"results/instance_perturbation/{dataset_name}/{model_name}/data"
    else:
        path = f"results/instance_perturbation/{dataset_name}/{model_name}/{rel_init}/data"

    labels = ["LRP-z$^+$", "LRP-$\gamma$", "LRP-$\\varepsilon$", "GradCAM", "gradient", "activation",
              "random"]

    _, _, n_classes = get_dataset(dataset_name=dataset_name).values()
    model = get_model(model_name=model_name, classes=n_classes)
    model.eval()
    layer_names = get_layer_names(model, [torch.nn.Conv2d])
    if "yolo" in model_name:
        layer_names = layer_names[::4]
    if "deeplab" in model_name:
        layer_names = layer_names[::3]
    print(" ".join(layer_names))

    layers = []
    data = []
    for layer_name in layer_names:
        try:
            fname = f"{path}/instance_perturbation_{layer_name}.pth" if not insertion else f"{path}/instance_perturbation_{layer_name}_insertion.pth"
            data.append(torch.load(fname))
            layers.append(layer_name)
        except:
            continue

    plt.figure(figsize=(5, 3), dpi=300)
    print("Data:", data)
    print("Type of data:", type(data))
    methods = [m for m in data[0] if m != "steps"]
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

        p, = plt.plot(layers, vals, '.-', label=f"{labels[k]} (${vals.mean():.2f}\\pm{err:.2f}$)", zorder=k / len(methods))
        plt.fill_between(layers, vals - valerrs, vals + valerrs, alpha=0.2, zorder=k - len(methods), color=p.get_color())

    plt.legend(fontsize="small")
    if model_name == "yolov5":
        mn = "YOLOv5"
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
    plt.title(f"{mn} - {ds}")
    plt.ylabel("AOC concept flipping" if not insertion else "AUC concept insertion")
    plt.xlabel("convolutional layer")
    plt.xticks(rotation=90)
    plt.xticks(layers, [get_layer_names(model, [torch.nn.Conv2d]).index(l) for l in layers])
    plt.tight_layout()
    fname = f"{path}/concept_perturbation_{model_name}_{dataset_name}.pdf" if not insertion else f"{path}/concept_perturbation_{model_name}_{dataset_name}_insertion.pdf"
    plt.savefig(fname, dpi=300, transparent=True)
    #plt.show()


if __name__ == "__main__":
    main()
