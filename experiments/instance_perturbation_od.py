import os
import random
import click
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from zennit.core import Composite
from datasets import get_dataset
from models import get_model

from utils.crp_configs import CANONIZERS

import torch.nn.functional as F

random.seed(10)


@click.command()

@click.option("--model_name", default="yolov5_nc7")
@click.option("--dataset_name", default="coco2017")
@click.option("--layer_name", default="model.10.conv")
@click.option("--num_samples", default=100)
@click.option("--batch_size", default=2)
@click.option("--insertion", default=False, type=bool)
def main(model_name, dataset_name, layer_name, num_samples, batch_size, insertion):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_dataset, n_classes = get_dataset(dataset_name=dataset_name).values()
    dataset = test_dataset()

    model = get_model(model_name=model_name, classes=n_classes)
    model_masked = get_model(model_name=model_name, classes=n_classes)
    model = model.to(device)
    model_masked = model_masked.to(device)
    model.eval()
    model_masked.eval()

    experiments = [{"label": "LRP-zplus",
                    "name": "crvs_zplus"},
                   {"label": "LRP-gamma",
                    "name": "crvs_gamma"},
                   {"label": "LRP-eps",
                    "name": "crvs_eps"},
                   {"label": "GradCAM",
                    "name": "crvs_gradcam"},
                   {"label": "Gradient",
                    "name": "crvs_grad"},
                   {"label": "activation",
                    "name": "cavs_max"},
                   {"label": "random",
                    "name": "random"},
                   ]

    for exp in experiments:
        exp['vecs'] = []
        exp['samples'] = []

    print("Loading concept vectors...")

    classes_unique = []
    for c in np.arange(0, n_classes):
        try:
            data = torch.load(
                f"results/global_class_concepts/{dataset_name}/{model_name}/logits/{layer_name}_class_{c}.pth")
            for exp in experiments:
                if exp["label"] != "random":
                    exp['vecs'].append(torch.stack(data[exp['name']], 0))
                    exp['samples'].append(data["samples"])
                else:
                    exp['vecs'].append(torch.rand_like(torch.stack(data[experiments[0]['name']], 0)))
                    exp['samples'].append(data["samples"])
            classes_unique.append(c)
        except:
            continue
    print(classes_unique)
    N = int(exp['vecs'][0].shape[-1])
    hooks = []
    neuron_indices = []

    if model_name == "yolov6":
        setattr(model.detect, "sm", torch.nn.Identity())
        setattr(model, "forward", model.forward_logits)
        setattr(model_masked.detect, "sm", torch.nn.Identity())
        setattr(model_masked, "forward", model_masked.forward_logits) #self.explainable_output
    if model_name == "yolov5":
        setattr(model.model[24], "sm", torch.nn.Identity())
        setattr(model, "forward", model.forward_logits)
        setattr(model_masked.model[24], "sm", torch.nn.Identity())
        setattr(model_masked, "forward", model_masked.forward_logits) #self.explainable_output

    if model_name == "yolov5_nc7":
        setattr(model.model[24], "sm", torch.nn.Identity())
        setattr(model, "forward", model.forward_logits)
        setattr(model_masked.model[24], "sm", torch.nn.Identity())
        setattr(model_masked, "forward", model_masked.forward_logits)  # self.explainable_output

    if model_name == "yolov5_nc7_attention":
        setattr(model.model[24], "sm", torch.nn.Identity())
        setattr(model, "forward", model.forward_logits)
        setattr(model_masked.model[24], "sm", torch.nn.Identity())
        setattr(model_masked, "forward", model_masked.forward_logits)  # self.explainable_output

    if model_name == "yolov5_nc7_with_semi_supervision":
        setattr(model.model[24], "sm", torch.nn.Identity())
        setattr(model, "forward", model.forward_logits)
        setattr(model_masked.model[24], "sm", torch.nn.Identity())
        setattr(model_masked, "forward", model_masked.forward_logits)  # self.explainable_output


    composite = Composite(canonizers=[CANONIZERS[model_name]()])

    prop_cycle = plt.rcParams['axes.prop_cycle']
    COLORS = prop_cycle.by_key()['color']

    plt.figure(dpi=300)
    steps = np.round(np.linspace(0, N, 15)).astype(int)
    steps = steps[1:] if not insertion else steps

    with composite.context(model_masked) as model_masked_mod:

        def hook(m, i, o):
            for b, batch_indices in enumerate(neuron_indices):
                if insertion:
                    indices_v = [x not in batch_indices for x in range(o.shape[1])]
                else:
                    indices_v = [x in batch_indices for x in range(o.shape[1])]
                o[b][indices_v] = o[b][indices_v] * 0

        for n, m in model_masked_mod.named_modules():
            if n == layer_name:
                hooks.append(m.register_forward_hook(hook))

        class1 = np.array([random.choice(classes_unique) for _ in range(num_samples)])
        samples = [experiments[0]["samples"][classes_unique.index(c)][list(np.where(class1 == c)[0]).index(j)] for j, c in enumerate(class1)]

        all_samples = np.array(samples)
        batches = int(np.ceil(len(all_samples) / batch_size))
        diffs_df = {}

        output = []
        data_ = []
        targets_ = []
        for b in range(batches):
            data = torch.stack([dataset[s][0] for s in all_samples[b * batch_size: (b + 1) * batch_size]])
            data = data.to(device)
            data_.append(data)
            out = model(data).detach().cpu()
            confidences = torch.nn.functional.sigmoid(out[..., 5:]) * torch.nn.functional.sigmoid(out[..., 4:5])

            t = [confidences[i, :, class1[b * batch_size: (b + 1) * batch_size][i]].argmax().item() for i in range(len(confidences))]
            targets_.append(t)
            output.append(torch.stack([out[i, t[i], 5 + class1[b * batch_size: (b + 1) * batch_size][i]] for i in range(len(confidences))]))


        for i, exp in enumerate(tqdm(experiments)):
            vecs = torch.stack([exp["vecs"][classes_unique.index(c)][list(np.where(class1 == c)[0]).index(j)] for j, c
                                in enumerate(class1)])
            diffs = []
            diffs_all = []
            topk1 = torch.topk(vecs, N)
            for k in steps:
                diff = []
                neuron_indices_all = topk1.indices[:, :k]
                for b in range(batches):
                    neuron_indices = neuron_indices_all[b * batch_size: (b + 1) * batch_size]
                    out = model_masked_mod(data_[b]).detach().cpu()
                    t = targets_[b]
                    out_diff = (torch.stack([out[i, t[i], 5 + class1[b * batch_size: (b + 1) * batch_size][i]] for i in range(len(t))]) - output[b])
                    diff.extend([o for o in out_diff.numpy()])
                diffs.append(np.mean(diff))
                diffs_all.append(diff)
                print(np.mean(diff), np.std(diff) / np.sqrt(len(diff)))
            diffs = np.array(diffs)
            diffs_all = np.array(diffs_all)
            plt.plot(steps, diffs, 'o--', label=exp["label"])
            diffs_df[exp["label"]] = diffs_all
            diffs_df["steps"] = steps

    plt.legend()
    plt.xlabel("flipped concepts")
    plt.ylabel("mean logit change")
    path = f"results/instance_perturbation/{dataset_name}/{model_name}"
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + "/data", exist_ok=True)
    if insertion:
        plt.savefig(f"{path}/instance_perturbation_{layer_name}_insertion.pdf", dpi=300, transparent=True)
        torch.save(diffs_df, f"{path}/data/instance_perturbation_{layer_name}_insertion.pth")
    else:
        plt.savefig(f"{path}/instance_perturbation_{layer_name}.pdf", dpi=300, transparent=True)
        torch.save(diffs_df, f"{path}/data/instance_perturbation_{layer_name}.pth")
    #plt.show()
    [hook.remove() for hook in hooks]
    print("debug")


if __name__ == "__main__":
    main()
