import os

import click
import numpy as np
import torch
from crp.helper import get_layer_names
from yolov5.cbam import CBAM
from tqdm import tqdm

from datasets import get_dataset
from models import get_model

from utils.crp import ChannelConcept
from utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS
from utils.zennit_composites import EpsilonPlusFlat, EpsilonGammaFlat, EpsilonFlat, GradientComposite


@click.command()
@click.option("--model_name", default="yolov5_nc7_attention")
@click.option("--dataset_name", default="coco2017")
@click.option("--class_id", default=6)
@click.option("--batch_size", default=8)
@click.option("--rel_init", default="logits", help="[ones, prob, logits]")
def main(model_name, dataset_name, class_id, batch_size, rel_init):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_dataset, n_classes = get_dataset(dataset_name=dataset_name).values()
    dataset = test_dataset()

    model = get_model(model_name=model_name, classes=n_classes)
    model = model.to(device)
    model.eval()

    attribution = ATTRIBUTORS[model_name](model)
    setattr(attribution, "rel_init", rel_init)
    lrp_zp_composite = EpsilonPlusFlat(canonizers=[CANONIZERS[model_name]()])
    lrp_gamma_composite = EpsilonGammaFlat(canonizers=[CANONIZERS[model_name]()])
    lrp_eps_composite = EpsilonFlat(canonizers=[CANONIZERS[model_name]()])
    grad_composite = GradientComposite(canonizers=[CANONIZERS[model_name]()])
    cc = ChannelConcept()
    condition = [{"y": class_id}]
    deleetd=0
    for ii,l in enumerate(list(model.model)):
        if isinstance(model.model[ii-deleetd], CBAM):
            del model.model[ii-deleetd]
            deleetd+=1
    layer_names = get_layer_names(model, [torch.nn.Conv2d])

    layer_map = {layer: cc for layer in layer_names}
    fv = VISUALIZATIONS[model_name](attribution,
                                    dataset,
                                    layer_map,
                                    preprocess_fn=lambda x: x,
                                    path=f"{model_name}_{dataset_name}",
                                    max_target="max")

    samples = np.array([i for i in range(len(dataset)) if class_id in fv.multitarget_to_single(fv.get_data_sample(i)[1])])
    n_samples = len(samples)
    n_batches = int(np.ceil(n_samples / batch_size))

    crvs_zplus = dict(zip(layer_names, [[] for _ in layer_names]))
    crvs_gamma = dict(zip(layer_names, [[] for _ in layer_names]))
    crvs_eps = dict(zip(layer_names, [[] for _ in layer_names]))
    cavs_max = dict(zip(layer_names, [[] for _ in layer_names]))
    cavs_mean = dict(zip(layer_names, [[] for _ in layer_names]))
    crvs_grad = dict(zip(layer_names, [[] for _ in layer_names]))
    crvs_gradcam = dict(zip(layer_names, [[] for _ in layer_names]))
    smpls = []

    for i in tqdm(range(n_batches)):
        samples_batch = samples[i * batch_size:(i + 1) * batch_size]
        data = torch.stack([dataset[j][0] for j in samples_batch], dim=0).to(device).requires_grad_()
        setattr(attribution, "rel_init", rel_init + "_zplus")
        attr_zplus = attribution(data, condition, lrp_zp_composite, record_layer=layer_names, init_rel=1)
        setattr(attribution, "rel_init", rel_init)
        attr_gamma = attribution(data, condition, lrp_gamma_composite, record_layer=layer_names, init_rel=1)
        attr_eps = attribution(data, condition, lrp_eps_composite, record_layer=layer_names, init_rel=1)
        setattr(attribution, "rel_init", rel_init + "_grad")
        attr_grad = attribution(data, condition, grad_composite, record_layer=layer_names, init_rel=1)
        setattr(attribution, "rel_init", rel_init)
        non_zero = (attr_zplus.heatmap.sum((1, 2)).abs() > 0).detach().cpu().numpy()
        samples_nz = samples_batch[non_zero]
        if samples_nz.size:
            smpls += [s for s in samples_nz]
            rels_zplus = [cc.attribute(attr_zplus.relevances[layer][non_zero], abs_norm=True) for layer in layer_names]
            rels_gamma = [cc.attribute(attr_gamma.relevances[layer][non_zero], abs_norm=True) for layer in layer_names]
            rels_eps = [cc.attribute(attr_eps.relevances[layer][non_zero], abs_norm=True) for layer in layer_names]
            acts_max = [attr_zplus.activations[layer][non_zero].flatten(start_dim=2).max(2)[0] for layer in layer_names]
            acts_mean = [attr_zplus.activations[layer][non_zero].mean((2, 3)) for layer in layer_names]
            gradient = [cc.attribute(attr_grad.relevances[layer][non_zero]) for layer in layer_names]
            gradcam = [cc.attribute(attr_zplus.activations[layer][non_zero].clamp(min=0)
                                    * attr_grad.relevances[layer][non_zero].mean((2, 3))[..., None, None])
                       for layer in layer_names]
            for l, r_zplus, r_gamma, r_eps, amax, amean, grad, gradc in zip(layer_names,
                                                  rels_zplus, rels_gamma, rels_eps,  acts_max, acts_mean, gradient, gradcam):
                crvs_zplus[l] += r_zplus.detach().cpu()
                crvs_gamma[l] += r_gamma.detach().cpu()
                crvs_eps[l] += r_eps.detach().cpu()
                cavs_max[l] += amax.detach().cpu()
                cavs_mean[l] += amean.detach().cpu()
                crvs_grad[l] += grad.detach().cpu()
                crvs_gradcam[l] += gradc.detach().cpu()

    path = f"results/global_class_concepts/{dataset_name}/{model_name}/{rel_init}"
    os.makedirs(path, exist_ok=True)
    torch.save({"samples": smpls,
                "crvs_zplus": crvs_zplus,
                "crvs_gamma": crvs_gamma,
                "crvs_eps": crvs_eps,
                "cavs_max": cavs_max,
                "cavs_mean": cavs_mean,
                "crvs_grad": crvs_grad,
                "crvs_gradcam": crvs_gradcam},
               f"{path}/class_{class_id}.pth")

    for layer in layer_names:
        torch.save({"samples": smpls,
                    # "output": output,
                    "crvs_zplus": crvs_zplus[layer],
                    "crvs_gamma": crvs_gamma[layer],
                    "crvs_eps": crvs_eps[layer],
                    "cavs_max": cavs_max[layer],
                    "cavs_mean": cavs_mean[layer],
                    "crvs_grad": crvs_grad[layer],
                   "crvs_gradcam": crvs_gradcam[layer]},
                   f"{path}/{layer}_class_{class_id}.pth")


if __name__ == "__main__":
    main()
