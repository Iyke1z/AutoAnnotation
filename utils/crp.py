import copy
import math
import os
from typing import List, Dict, Union, Callable, Tuple
from pathlib import Path
import numpy as np
import torch
from crp.attribution import CondAttribution, attrResult
from crp.concepts import Concept
from crp.concepts import ChannelConcept as ChannelConcept_
from crp.helper import load_maximization, load_statistics
from crp.hooks import FeatVisHook
from crp.maximization import Maximization as Maximization_
from crp.statistics import Statistics as Statistics_
from crp.visualization import FeatureVisualization
from tqdm import tqdm
from zennit.composites import NameMapComposite
from zennit.core import Composite

class Statistics(Statistics_):
    def analyze_layer(self, d_c_sorted, rel_c_sorted, rf_c_sorted, layer_name, targets):
        t_unique = torch.unique(targets)
        for t in t_unique:
            t_indices = targets.t() == t
            num_channels = targets.shape[1]
            d_c_t = d_c_sorted.t()[t_indices].view(num_channels, -1).t()
            rel_c_t = rel_c_sorted.t()[t_indices].view(num_channels, -1).t()
            rf_c_t = rf_c_sorted.t()[t_indices].view(num_channels, -1).t()
            self.concatenate_with_results(layer_name, int(t), d_c_t, rel_c_t, rf_c_t)
            self.sort_result_array(layer_name, int(t))


import os


class Maximization(Maximization_):
    def collect_results(self, path_list: List[str], d_index: Tuple[int, int] = None):

        self.delete_result_arrays()

        pbar = tqdm(total=len(path_list), dynamic_ncols=True)

        for path in path_list:
            filename = path.split("/")[-1]
            ending = filename.split(".")[-1].split("_")
            ending_beginning = ending[0]
            ending = [x for x in ending[1:] if not x.isnumeric()]
            ending = [x for x in ending if x != ""]
            if len(ending):
                ending = "_" + "_".join(ending)
            else:
                ending = ""
            l_name = ".".join(filename.split(".")[:-1]) + f".{ending_beginning}{ending}"

            save_path = self.PATH / Path(filename + "data.npy")
            save_path = str(save_path).replace('yolov5_coco2017\\RelMax_sum_normed\\yolov5_coco2017\\RelMax_sum_normed',
                                               'yolov5_coco2017\\RelMax_sum_normed')

            # Print out the path for debugging
            print("Saving to path:", save_path)
            print(list(self.d_c_sorted.keys()))
            print("l_name:", l_name)

            # Ensure directory exists
            directory = os.path.dirname(save_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

            #np.save(save_path, self.d_c_sorted[l_name].cpu().numpy())

            d_c_sorted = np.load(path + "data.npy")
            rf_c_sorted = np.load(path + "rf.npy")
            rel_c_sorted = np.load(path + "rel.npy")

            d_c_sorted, rf_c_sorted, rel_c_sorted = map(torch.from_numpy, [d_c_sorted, rf_c_sorted, rel_c_sorted])

            self.concatenate_with_results(l_name, d_c_sorted, rel_c_sorted, rf_c_sorted)
            self.sort_result_array(l_name)

            pbar.update(1)

        for path in path_list:
            for suffix in ["data.npy", "rf.npy", "rel.npy"]:
                os.remove(path + suffix)

        pbar.close()

        result = self._save_results(d_index)

        return result

    def analyze_layer(self, rel, concept: Concept, layer_name: str, data_indices):

        argsort, rel_c_sorted, rf_c_sorted = concept.reference_sampling(
            rel, layer_name, self.max_target, self.abs_norm)
        # convert batch index to dataset wide index
        data_indices = torch.from_numpy(data_indices).to(argsort)
        d_c_sorted = torch.take(data_indices, argsort)

        SZ = self.SAMPLE_SIZE
        self.concatenate_with_results(layer_name, d_c_sorted[:SZ], rel_c_sorted[:SZ], rf_c_sorted[:SZ])
        self.sort_result_array(layer_name)

        return d_c_sorted, rel_c_sorted, rf_c_sorted, argsort


class ChannelConcept(ChannelConcept_):
    def reference_sampling(self, relevance, layer_name: str = None, max_target: str = "sum", abs_norm=True):
        """
        Parameters:
            max_target: str. Either 'sum' or 'max'.
        """

        # position of receptive field neuron
        rel_l = relevance.view(*relevance.shape[:2], -1)
        rf_neuron = torch.argmax(rel_l, dim=-1)

        # channel maximization target
        if max_target == "sum":
            rel_l = torch.sum(relevance.view(*relevance.shape[:2], -1), dim=-1)

        elif max_target == "max":
            rel_l = torch.max(rel_l, dim=2)[0]

        else:
            raise ValueError("<max_target> supports only 'max' or 'sum'.")

        if abs_norm:
            rel_l = rel_l / (torch.abs(rel_l).sum(-1).view(-1, 1) + 1e-10)

        d_ch_sorted = torch.argsort(rel_l, dim=0, descending=True)
        rel_ch_sorted = torch.gather(rel_l, 0, d_ch_sorted)
        rf_ch_sorted = torch.gather(rf_neuron, 0, d_ch_sorted)

        return d_ch_sorted, rel_ch_sorted, rf_ch_sorted


class CondAttributionDiff(CondAttribution):
    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        self.model_copy = copy.deepcopy(model)
        self.take_prediction = 0

    def __call__(
            self, data: torch.tensor, conditions: List[Dict[str, List]],
            composite: Composite = None, record_layer: List[str] = [],
            mask_map: Union[Callable, Dict[str, Callable]] = ChannelConcept.mask, start_layer: str = None,
            init_rel=None,
            on_device: str = None) -> attrResult:
        # with condition on layer (conditional relevance and relevance of other paths)

        attr_result = CondAttribution.__call__(self, copy.copy(data), conditions, composite, record_layer, mask_map,
                                               start_layer, init_rel, on_device)
        multiple_conditions = [condition for condition in conditions if (len(condition) > 1)]
        if not multiple_conditions:
            return attr_result

        # for conditonal relevances compute relevance of other paths
        start_layer_ = "y" if not start_layer else start_layer
        [condition.update(dict(
            zip(
                [k for k in condition.keys() if (k != start_layer_)],
                [[] for k, v in condition.items() if (k != start_layer_)]
            ))) for condition in multiple_conditions]
        model = self.model
        self.model = self.model_copy
        attr_result_ = CondAttribution.__call__(self, copy.copy(data), multiple_conditions, composite, record_layer,
                                                mask_map, start_layer, init_rel, on_device)
        self.model = model
        # compute relevance differences to only get conditional relevance
        mult_conditions = [(len(condition) > 1) for condition in conditions]
        attr_result.heatmap[mult_conditions] -= attr_result_.heatmap
        # for k in attr_result.relevances.keys():
        #     attr_result.relevances[k][mult_conditions] -= attr_result_.relevances[k] TODO: why does this not work?
        return attr_result


class CondAttributionLocalization(CondAttributionDiff):
    def backward_initialization(self, prediction, target_list, init_rel, layer_name, retain_graph=False):

        if target_list:
            r = torch.zeros_like(prediction).to(self.device)
            for i, target in enumerate(target_list):
                if prediction[i].shape[0] == 0:
                    print("no predicted boxes")
                else:
                    k = min(self.take_prediction + 1, prediction[i].shape[0])
                    best_bb_id = torch.topk(prediction[i], k, dim=0).indices[k - 1, target].item()
                    if self.take_prediction != 0:
                        print("taking prediction num. ", k - 1, " (wanted ", self.take_prediction, ")")
                    r[i, best_bb_id, target] = torch.ones_like(r[i, best_bb_id, target]) * (
                            prediction[i, best_bb_id, target] > 0.25)
            init_rel = r / (r.sum() + 1e-12)
        else:
            prediction = prediction.clamp(min=0)

        return CondAttributionDiff.backward_initialization(self, prediction, None, init_rel, layer_name, retain_graph)

class CondAttributionSegmentation(CondAttributionDiff):

    def __init__(self, model: torch.nn.Module):

        super().__init__(model)
        self.mask = 1
        self.rel_init = "logits"

    def backward_initialization(self, prediction, target_list, init_rel, layer_name, retain_graph=False):

        if target_list:
            r = torch.zeros_like(prediction).to(self.device)
            pred = torch.nn.functional.softmax(prediction, dim=1)
            for i, target in enumerate(target_list):
                if isinstance(target, List):
                    assert len(target) == 1
                    target = target[0]
                argmax = (torch.argmax(pred, dim=1) == target)[i]
                expl = prediction
                if "zplus" in self.rel_init:
                    expl = expl.clamp(min=0)
                if "ones" in self.rel_init:
                    expl = torch.ones_like(expl)
                elif "prob" in self.rel_init:
                    expl = pred
                if "grad" in self.rel_init:
                    expl = expl / (prediction + 1e-10)
                r[i, target, :, :] = expl[i, target, :, :] * argmax
                r = r * self.mask
            init_rel = r / (r.sum() + 1e-12)
        else:
            prediction = prediction.clamp(min=0)

        return CondAttributionDiff.backward_initialization(self, prediction, None, init_rel, layer_name, retain_graph)


class FeatureVisualizationMultiTarget(FeatureVisualization):

    def __init__(self, attribution: CondAttribution, dataset, layer_map: Dict[str, Concept],
                 preprocess_fn: Callable = None, max_target="sum", abs_norm=True, path="FeatureVisualization",
                 device=None):

        super().__init__(attribution, dataset, layer_map, preprocess_fn, max_target, abs_norm, path, device)

        self.RelMax = Maximization("relevance", "sum", abs_norm, path)
        self.ActMax = Maximization("activation", max_target, abs_norm, path)
        self.RelStats = Statistics("relevance", "sum", abs_norm, path)
        self.ActStats = Statistics("activation", max_target, abs_norm, path)

        self.ReField = None

    def multitarget_to_single(self, multi_target):
        multi_target = np.array(multi_target).astype(int)
        single_targets = np.argwhere(multi_target == 1).reshape(-1)
        assert len(single_targets) == np.sum(multi_target)
        return single_targets

    def get_max_reference(
            self, concept_ids: list, layer_name: str, mode="relevance", r_range: Tuple[int, int] = (0, 8),
            heatmap=False, composite=None, batch_size=2, rf=True):
        """
        Retreive reference samples for a list of concepts in a layer. Relevance and Activation Maximization
        are availble if FeatureVisualization was computed for the mode. If the ReceptiveField of the layer
        was computed, it can be used to cut out the most representative part of the sample. In addition,
        conditional heatmaps can be computed on reference samples.

        Parameters:
        ----------
            concept_ids: list
            layer_name: str
            mode: "relevance" or "activation"
                Relevance or Activation Maximization
            r_range: Tuple(int, int)
                Aange of N-top reference samples. For example, (3, 7) corresponds to the Top-3 to -6 samples.
                Argument must be a closed set i.e. second element of tuple > first element.
            heatmap: boolean
                If True, compute conditional heatmaps on reference samples. Please make sure to supply a composite.
            composite: zennit.composites or None
                If `heatmap` is True, `composite` is used for the CondAttribution object.
            batch_size: int
                If heatmap is True, describes maximal batch size of samples to compute for conditional heatmaps.
            rf: boolean
                If True, crop samples or heatmaps with receptive field using the `weight_receptive_field` method.

        Returns:
        -------
            ref_c: dictionary.
                Key values correspond to channel index and values are reference samples.
                If rf is True, reference samples are a list of torch.Tensor with different shapes. Otherwise the
                dictionary values are torch.Tensor with same shape.
        """

        ref_c = {}

        if mode == "relevance":
            d_c_sorted, rel_c_sorted, rf_c_sorted = load_maximization(self.RelMax.PATH, layer_name)
        elif mode == "activation":
            d_c_sorted, rel_c_sorted, rf_c_sorted = load_maximization(self.ActMax.PATH, layer_name)
        else:
            raise ValueError("`mode` must be `relevance` or `activation`")

        for c_id in concept_ids:

            d_indices = d_c_sorted[r_range[0]:r_range[1], c_id]
            r_values = rel_c_sorted[r_range[0]:r_range[1], c_id]

            if heatmap:
                data_batch, targets_multi = self.get_data_concurrently(d_indices, preprocessing=True)
                if mode == "relevance":
                    targets_single = []
                    for i_t, target in enumerate(targets_multi):
                        single_targets = self.multitarget_to_single(target)
                        for st in single_targets:
                            targets_single.append(st)

                    targets = np.zeros(r_range[1] - r_range[0]).astype(int)
                    for t in np.arange(81): # TODO: make more general (here 80 classes correspond to COCO classes)
                        try:
                            target_stats = load_statistics(self.RelStats.PATH, layer_name, t)
                            td_indices = target_stats[0][:, c_id]
                            tr_values = target_stats[1][:, c_id]
                            cond = [True if (x in td_indices) and (tr_values[list(td_indices).index(x)] == r) else False
                                    for x, r in
                                    zip(d_indices, r_values)]
                            targets[cond] = int(t)
                        except FileNotFoundError:
                            continue
                    data_batch = self.attribution_on_reference(data_batch, c_id, layer_name, composite, batch_size,
                                                               targets)
                else:
                    data_batch = self.attribution_on_reference(data_batch, c_id, layer_name, composite, batch_size)
            else:
                data_batch, _ = self.get_data_concurrently(d_indices, preprocessing=False)

            if rf and self.ReField:
                neuron_ids = rf_c_sorted[r_range[0]:r_range[1], c_id]
                data_batch = self.ReField.weight_receptive_field(neuron_ids, data_batch, layer_name)

            ref_c[c_id] = data_batch

        return ref_c

    def attribution_on_reference(self, data, concept_id: int, layer_name: str, composite, batch_size=2, targets=None):

        n_samples = len(data)
        if n_samples > batch_size:
            batches = math.ceil(n_samples / batch_size)
        else:
            batches = 1
            batch_size = n_samples

        heatmaps = []
        for b in range(batches):
            data_batch = data[b * batch_size: (b + 1) * batch_size]

            if targets is None:
                conditions = [{layer_name: [concept_id]}]
                start_layer = layer_name
            else:
                targets_batch = targets[b * batch_size: (b + 1) * batch_size]
                conditions = [{layer_name: [concept_id], "y": t} for t in targets_batch]
                ## THIS OVERWRITES FOR OTHER BATCHES
                # layer_name = None
                start_layer = None
            attr = self.attribution(data_batch, conditions, composite, start_layer=start_layer)
            heatmaps.append(attr.heatmap)

        return torch.cat(heatmaps, dim=0)

    def run_distributed(self, composite: Composite, data_start, data_end, batch_size=2, checkpoint=500, on_device=None):
        """
        max batch_size = max(multi_targets) * data_batch
        data_end: exclusively counted
        """

        self.saved_checkpoints = {"r_max": [], "a_max": [], "r_stats": [], "a_stats": []}
        last_checkpoint = 0

        n_samples = data_end - data_start
        samples = np.arange(start=data_start, stop=data_end)

        if n_samples > batch_size:
            batches = math.ceil(n_samples / batch_size)
        else:
            batches = 1
            batch_size = n_samples

        # feature visualization is performed inside forward and backward hook of layers
        name_map, dict_inputs = [], {}
        for l_name, concept in self.layer_map.items():
            hook = FeatVisHook(self, concept, l_name, dict_inputs, on_device)
            name_map.append(([l_name], hook))
        fv_composite = NameMapComposite(name_map)

        if composite:
            composite.register(self.attribution.model)
        fv_composite.register(self.attribution.model)

        pbar = tqdm(total=batches, dynamic_ncols=True)

        for b in range(batches):

            pbar.update(1)
            samples_batch = samples[b * batch_size: (b + 1) * batch_size]
            data_batch, targets_samples = self.get_data_concurrently(samples_batch, preprocessing=True)

            targets_samples = np.array(targets_samples)  # numpy operation needed

            # convert multi target to single target if user defined the method
            data_broadcast, targets, sample_indices = [], [], []
            try:
                for i_t, target in enumerate(targets_samples):
                    single_targets = self.multitarget_to_single(target)
                    for st in single_targets:
                        targets.append(st)
                        data_broadcast.append(data_batch[i_t])
                        sample_indices.append(samples_batch[i_t])
                if len(data_broadcast) == 0:
                    continue
                # TODO: test stack
                data_broadcast = torch.stack(data_broadcast, dim=0)
                sample_indices = np.array(sample_indices)
                targets = np.array(targets)

            except NotImplementedError:
                data_broadcast, targets, sample_indices = data_batch, targets_samples, samples_batch

            conditions = [{self.attribution.MODEL_OUTPUT_NAME: [t]} for t in targets]

            if n_samples > batch_size:
                batches_ = math.ceil(len(conditions) / batch_size)
            else:
                batches_ = 1

            for b_ in range(batches_):
                data_broadcast_ = data_broadcast[b_ * batch_size: (b_ + 1) * batch_size]
                # print(len(conditions), len(data_broadcast_))
                conditions_ = conditions[b_ * batch_size: (b_ + 1) * batch_size]
                # dict_inputs is linked to FeatHooks
                dict_inputs["sample_indices"] = sample_indices[b_ * batch_size: (b_ + 1) * batch_size]
                dict_inputs["targets"] = targets[b_ * batch_size: (b_ + 1) * batch_size]

                self.attribution(data_broadcast_, conditions_, None)

            if b % checkpoint == checkpoint - 1:
                self._save_results((last_checkpoint, sample_indices[-1] + 1))
                last_checkpoint = sample_indices[-1] + 1

        # TODO: what happens if result arrays are empty?
        self._save_results((last_checkpoint, sample_indices[-1] + 1))

        if composite:
            composite.remove()
        fv_composite.remove()

        pbar.close()

        return self.saved_checkpoints

    @torch.no_grad()
    def analyze_relevance(self, rel, layer_name, concept, data_indices, targets):
        """
        Finds input samples that maximally activate each neuron in a layer and most relevant samples
        """
        # TODO: dummy target for extra dataset
        d_c_sorted, rel_c_sorted, rf_c_sorted, argsort = self.RelMax.analyze_layer(rel, concept, layer_name, data_indices)

        targets = torch.take(torch.Tensor(targets).to(argsort), argsort)
        self.RelStats.analyze_layer(d_c_sorted, rel_c_sorted, rf_c_sorted, layer_name, targets)

    @torch.no_grad()
    def analyze_activation(self, act, layer_name, concept, data_indices, targets):
        """
        Finds input samples that maximally activate each neuron in a layer and most relevant samples
        """

        # activation analysis once per sample if multi target dataset
        unique_indices = np.unique(data_indices, return_index=True)[1]
        data_indices = data_indices[unique_indices]
        act = act[unique_indices]
        targets = targets[unique_indices]

        d_c_sorted, act_c_sorted, rf_c_sorted, argsort = self.ActMax.analyze_layer(act, concept, layer_name, data_indices)

        targets = torch.take(torch.Tensor(targets).to(argsort), argsort)
        self.ActStats.analyze_layer(d_c_sorted, act_c_sorted, rf_c_sorted, layer_name, targets)

class FeatureVisualizationLocalization(FeatureVisualizationMultiTarget):
    def get_data_sample(self, index, preprocessing=True) -> Tuple[torch.tensor, List[int]]:
        """
        returns a data sample from dataset at index.

        Parameter:
            index: integer
            preprocessing: boolean.
                If True, return the sample after preprocessing. If False, return the sample for plotting.
        """

        data, target = self.dataset[index]
        target = target[..., 1].long()
        targets = np.unique(target)
        targets = np.random.permutation(targets)[:min(len(targets), 99)]
        if not len(targets):
            targets = np.array([1])
        else:
            targets = [1 if (i in targets.astype(int)) else 0 for i in range(targets.max() + 1)]
        return data.unsqueeze(0).to(self.device).requires_grad_(), targets

class FeatureVisualizationSegmentation(FeatureVisualizationMultiTarget):
    def get_data_sample(self, index, preprocessing=True) -> Tuple[torch.tensor, List[int]]:
        """
        returns a data sample from dataset at index.

        Parameter:
            index: integer
            preprocessing: boolean.
                If True, return the sample after preprocessing. If False, return the sample for plotting.
        """

        data, target = self.dataset[index]
        targets = torch.unique(target)
        targets = torch.Tensor([t for t in targets if t != 255]).int() # no background class
        targets = targets[torch.randperm(len(targets))][:min(len(targets) + 1, 99)]

        # print(data.shape)
        targets = [1 if (i in list(targets.numpy())) else 0 for i in range(targets.max() + 1)]
        targets = np.array(targets).flatten().astype(int)
        return data.unsqueeze(0).to(self.device).requires_grad_(), targets
