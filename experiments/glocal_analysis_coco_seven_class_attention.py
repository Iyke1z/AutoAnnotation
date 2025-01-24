import click
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.dirname(__file__)+"/../"))
from crp.helper import get_layer_names
from datasets import get_dataset
from models import get_model
from utils.crp import ChannelConcept
from utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES

torch.set_grad_enabled(True)  # Context-manager
@click.command()
@click.option("--model_name", default="yolov5_nc7_attention")
@click.option("--dataset_name", default="coco2017")
@click.option("--batch_size", default=4)
def main(model_name, dataset_name, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_dataset, n_classes = get_dataset(dataset_name=dataset_name).values()
    dataset = test_dataset(preprocessing=False)
    model = get_model(model_name=model_name, classes=n_classes)

    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])

    model = model.to(device)
    model.eval()
    cc = ChannelConcept()
    layer_names = get_layer_names(model, [torch.nn.Conv2d])
    layer_map = {layer: cc for layer in layer_names}

    attribution = ATTRIBUTORS[model_name](model)

    fv = VISUALIZATIONS[model_name](attribution,
                                    dataset,
                                    layer_map,
                                    preprocess_fn=lambda x: x,
                                    path=f"{model_name}_{dataset_name}",
                                    max_target="max")

    fv.run(composite, 0, len(dataset), batch_size, 100)


if __name__ == "__main__":
    main()
