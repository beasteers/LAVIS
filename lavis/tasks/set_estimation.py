"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
import os
from collections import defaultdict
from re import L
import numpy as np
import torch
# from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
from lavis.common.registry import registry
import lavis.common.dist_utils as dist_utils
from lavis.common.vqa_tools.vqa_clean import VQACleaner
from lavis.tasks.base_task import BaseTask
from lavis.common.predicate_utils.predicates import Predicate

import wandb


CLEAN = VQACleaner()

@registry.register_task("ekos")
class EpicKitchensTask(BaseTask):
    clean = CLEAN
    TRUTH = {
        'true': 'yes',
        'false': 'no',
        'y': 'yes',
        'n': 'no',
    }

    def __init__(self, num_beams, max_len, min_len):
        super().__init__()
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        return cls(
            num_beams=run_cfg.num_beams,
            max_len=run_cfg.max_len,
            min_len=run_cfg.min_len,
        )

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)
        # for name, dataset in datasets.items():
        #     dataset
        first = datasets[list(datasets)[0]]
        self.classes = first['val'].classes
        self.class_mismatch = False
        return datasets

    def valid_step(self, model, samples):
        results = []

        answer_pred = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )
        # answer_pred = np.array(['']*len(samples['image']))

        cls_pred = None
        if not self.class_mismatch and samples.get('targets') is not None and getattr(model, 'has_classifier', False):
            cls_pred = model.class_head(samples)
            if cls_pred.shape[1] != samples['targets'].shape[1]:
                self.class_mismatch = True
                cls_pred = None
        with open(f'{registry.get_path("result_dir")}/result_stream.txt', 'a') as fh:
            for i in range(len(answer_pred)):
                # print(samples["narration_id"][i])
                r = {
                    "question": samples["text_input"][i],
                    "answer_pred": answer_pred[i], 
                    "answer_true": samples["text_output"][i],
                    "image_id": samples["image_id"][i], 
                    "narration_id": samples["narration_id"][i], 
                    "noun": samples["noun"][i], 
                }
                if cls_pred is not None:
                    r.update({
                        "cls_pred": cls_pred[i].cpu().numpy().tolist(),
                        "cls_true": samples['targets'][i].cpu().numpy().tolist(),
                        # "cls_labels": samples['class_labels'][i].cpu().numpy().tolist(),
                    })
                results.append(r)
                fh.write(f'{json.dumps(r)}\n')
                if samples['sample_id'][i] in self.sample_index:
                    print("Sample:", samples['sample_id'][i], samples["narration_id"][i], samples["narration"][i])
                    print("in:", samples['text_input'][i])
                    print("pred:", answer_pred[i])
                    print("true:", samples["text_output"][i])
                    if cls_pred is not None:
                        cs = np.array(self.classes)
                        yt=samples['targets'][i].cpu().numpy()
                        yp=cls_pred[i].cpu().numpy()
                        print(cs[yt!=-1])
                        print(yt[yt!=-1])
                        print(np.round(yp[yt!=-1], 3))
                    self.result_table.add_data(
                        samples['sample_id'][i],
                        wandb.Video(norm_video(samples["image"][i]).cpu().numpy(), fps=3) 
                        if samples["image"].ndim == 5 else
                        wandb.Image(samples["image"][i].cpu()),
                        samples["text_input"][i],  # question
                        answer_pred[i],  # Predicted answer
                        samples["text_output"][i],  # True answer
                        '\n'.join(f'{c}[{t:.0f}]: {p:.3f}' for c, t, p in zip(cs[yt!=-1], yt[yt!=-1], yp[yt!=-1]))
                        if cls_pred is not None else "",  # cls prediction
                        samples["narration_id"][i],  # Narration ID
                        samples["narration"][i]  # Narration text
                    )
        
        return results

    def before_evaluation(self, model, dataset, **kwargs):
        print("before eval",kwargs)
        super().before_evaluation(model, dataset, **kwargs)
        self.sample_index = np.random.choice(len(dataset), min(60, len(dataset)), replace=False)
        print("eval samples:", self.sample_index, len(dataset))
        self.result_table = wandb.Table(columns=["index", "image", "question", "answer_pred", "answer_true", "cls", "narration_id", "narration"])

    def after_evaluation(self, val_result, split_name, epoch='?', **kwargs):
        # Log the table
        print("write table", len(self.result_table.data))
        wandb.log({"predictions": self.result_table})

        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_epic_kitchens_result_{epoch}",
            remove_duplicate="", 
        )
        try:
            os.rename(f'{registry.get_path("result_dir")}/result_stream.txt', f'{registry.get_path("result_dir")}/{split_name}_result_stream_{epoch}.txt')
        except Exception as e: print(e)
        # metrics = None
        # if split_name == 'val':
        metrics = self._report_metrics(result_file=result_file, split=split_name)
        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        results = json.load(open(result_file, "r"))

        # calculate metrics
        errors = []
        acc = []
        for res in results:
            try:
                acc.append(jaccard_score(res["answer_pred"], res["answer_true"]))
            except Exception as e:
                print("Could not parse", res["answer_pred"], e)
                errors.append(res["answer_pred"])
                acc.append(0)
        
        cls_metrics = {}
        if results[0].get('cls_true') is not None:
            try:
                y_true = np.array([d['cls_true'] for d in results]).astype(int)
                y_pred = np.array([d['cls_pred'] for d in results]).astype(float)
                if y_true.shape[1] == y_pred.shape[1]:
                    if split == 'val':
                        cls_metrics.update(compute_cls_metrics(y_true, y_pred))
                    else:
                        cls_metrics.update(compute_metrics_with_partial_ground_truth(y_true, y_pred, threshold=0.5, prefix='cls_'))
                    plot_ml_cm(y_true, y_pred, self.classes)
            except Exception:
                import traceback
                traceback.print_exc()
        if split == 'test':
            try:
                txt_true = [d['answer_true'] for d in results]
                txt_pred = [d['answer_pred'] for d in results]
                y_true, y_pred, lm_classes = convert_text_to_manyhot(txt_true, txt_pred)
                cls_metrics.update(compute_metrics_with_partial_ground_truth(y_true, y_pred, threshold=0.5, classes=lm_classes, prefix='lm_'))
                plot_ml_cm(y_true, y_pred, lm_classes, prefix='lm_')
            except Exception:
                import traceback
                traceback.print_exc()

        # report metrics
        accuracy = np.mean(acc)
        metrics = {"agg_metrics": accuracy, "accuracy": accuracy, "split": split, **cls_metrics}

        with open(os.path.join(registry.get_path("output_dir"), f"log.txt"), "a") as f:
            f.write(json.dumps(metrics) + "\n")
        with open(os.path.join(registry.get_path("output_dir"), f"parse_error.txt"), "a") as f:
            for x in errors:
                f.write(f'{x}\n')

        logging.info(metrics)

        return metrics
    
    def _jaccard(self, y_pred, y_true):
        # XXX: repetition/contradiction
        y_pred = {self.clean(x) for x in split_text(y_pred, '.', ',', n=-1)}
        y_true = {self.clean(x) for x in split_text(y_true, '.', ',', n=-1)}
        return len(y_pred & y_true) / (len(y_pred | y_true) or 1)





def split_text(text, *delims, n=-1):
    for d in delims:
        if d in text:
            return [y.strip() for y in text.split(d, n) if y.strip()]
    return ([text] + ['']*n) if n else [text]


def norm_video(img):
    img = img.float()
    for t in img:
        low, high = float(t.min()), float(t.max())
        t.clamp_(min=low, max=high)
        t.sub_(low).div_(max(high - low, 1e-5))
    img = img.mul(255).clamp(0, 255).byte()
    return img


 
def jaccard_score(y_pred, y_true):
    # XXX: repetition/contradiction
    y_pred = {CLEAN(x) for x in split_text(y_pred, '.', ',', n=-1)}
    y_true = {CLEAN(x) for x in split_text(y_true, '.', ',', n=-1)}
    return len(y_pred & y_true) / (len(y_pred | y_true) or 1)




def compute_cls_metrics(y_true, y_pred, threshold=0.5):
    y_true = y_true.astype(int)
    y_pred = (y_pred > threshold).astype(int)
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    y_pred = y_pred[y_true != -1]
    y_true = y_true[y_true != -1]
    return {
        'cls_acc': accuracy_score(y_true, y_pred), 
        'cls_f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=1), 
        'cls_f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=1),
    }


def convert_text_to_manyhot(txt_true, txt_pred):
    set_true = [{CLEAN(x) for x in split_text(t, '.', ',')} for t in txt_true]
    set_pred = [{CLEAN(x) for x in split_text(t, '.', ',')} for t in txt_pred]
    labels = sorted(
        {value for s in set_true for value in s} | 
        {value for s in set_pred for value in s}
    )
    y_true = np.array([[l in s for l in labels] for s in set_true], dtype=int)
    y_pred = np.array([[l in s for l in labels] for s in set_pred], dtype=int)
    return y_true, y_pred, labels


def compute_metrics_with_partial_ground_truth(y_true, y_pred_logits, threshold=0.5, include_per_class=False, classes=None, prefix=''):
    # Convert logits to binary predictions based on threshold
    y_pred = (y_pred_logits > threshold).astype(int)
    
    # Initialize containers for aggregated metrics
    precision_list, recall_list, f1_list = [], [], []
    total_true_positives, total_false_positives, total_false_negatives, total_correct, total_valid = 0, 0, 0, 0, 0
    
    # Iterate over each class to compute metrics, excluding -1 in y_true
    for i in range(y_true.shape[1]):
        valid_indices = y_true[:, i] != -1
        y_true_filtered = y_true[valid_indices, i]
        y_pred_filtered = y_pred[valid_indices, i]
        y_pred_filtered = y_pred_filtered > y_pred_filtered.mean()

        # Compute class-wise metrics
        precision = precision_score(y_true_filtered, y_pred_filtered, zero_division=0)
        recall = recall_score(y_true_filtered, y_pred_filtered, zero_division=0)
        f1 = f1_score(y_true_filtered, y_pred_filtered, zero_division=0)
        
        # Aggregate for macro average calculation
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        
        # Aggregate counts for micro average calculation
        tp = (y_pred_filtered & y_true_filtered).sum()
        fp = (y_pred_filtered & ~y_true_filtered).sum()
        fn = (~y_pred_filtered & y_true_filtered).sum()
        correct = (y_pred_filtered == y_true_filtered).sum()
        
        total_true_positives += tp
        total_false_positives += fp
        total_false_negatives += fn
        total_correct += correct
        total_valid += len(y_true_filtered)
    
    # Compute micro averages
    precision_micro = total_true_positives / max(1, total_true_positives + total_false_positives)
    recall_micro = total_true_positives / max(1, total_true_positives + total_false_negatives)
    f1_micro = 2 * (precision_micro * recall_micro) / max(1, precision_micro + recall_micro)
    accuracy = total_correct / max(1, total_valid)
    
    # Compute macro averages
    precision_macro = np.mean(precision_list)
    recall_macro = np.mean(recall_list)
    f1_macro = np.mean(f1_list)
    
    # Compile results
    per_class = {}
    if include_per_class:
        for i, (prec, rec, f1) in enumerate(zip(precision_list, recall_list, f1_list)):
            c = classes[i] if classes is not None else i
            if '(' in str(c):
                c = Predicate(c).name
            per_class[f'{prefix}class_{c}_precision'] = prec
            per_class[f'{prefix}class_{c}_recall'] = rec
            per_class[f'{prefix}class_{c}_f1'] = f1
    metrics = {
        f'{prefix}accuracy': accuracy,
        f'{prefix}precision_micro': precision_micro,
        f'{prefix}recall_micro': recall_micro,
        f'{prefix}f1_micro': f1_micro,
        f'{prefix}precision_macro': precision_macro,
        f'{prefix}recall_macro': recall_macro,
        f'{prefix}f1_macro': f1_macro,
        **per_class,
    }
    
    return metrics



def plot_ml_cm(y_true, y_pred, labels, ncols=8, threshold=0.5, s=3, prefix=''):
    # Compute multilabel confusion matrix
    y_pred = (y_pred > threshold).astype(int)
    # mask = y_true == -1
    # y_true_masked = np.where(mask, y_pred, y_true)
    # mcm = multilabel_confusion_matrix(y_true_masked, y_pred)
    # mcm = np.stack([
    #     np.stack([(y_true == 0) & (y_pred == 0), (y_true == 0) & (y_pred == 1)], axis=1),
    #     np.stack([(y_true == 1) & (y_pred == 0), (y_true == 1) & (y_pred == 1)], axis=1),
    # ], axis=1)
    mcm = np.zeros((len(labels), 2, 2))
    for yt, yp in zip(y_true, y_pred):
        for j, (yti, ypi) in enumerate(zip(yt, yp)):
            if yti != -1:
                # print(labels[j], yti, ypi)
                mcm[j, yti, ypi] += 1
    mcm = mcm.astype(float) / np.maximum(1, mcm.sum(-1, keepdims=True))

    # return mcm

    # Plotting the confusion matrices for each label
    nrows = int(np.ceil(len(mcm)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(s * ncols, s * nrows))
    for i, (label, ax) in enumerate(zip(labels, axes.flat)):
        # Confusion matrix for each label
        cm = mcm[i]
        
        # Display the confusion matrix
        cax = ax.matshow(cm, cmap='bone_r', vmin=0, vmax=1)
        # fig.colorbar(cax, ax=ax)
        
        # Annotate the matrix with text
        for (j, k), val in np.ndenumerate(cm):
            ax.text(k, j, f'{val:.0%}', ha='center', va='center', color='red')

        # Set labels and titles
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'{label}')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
    for ax in list(axes.flat)[len(labels):]:
        ax.remove()

    # fig.colorbar(im, ax=axes.ravel().tolist())
    plt.tight_layout()

    plt.savefig(f"{prefix}confusion_matrix.png")
    wandb.log({f"{prefix}confusion_matrix": wandb.Image(f"{prefix}confusion_matrix.png")})
    plt.close()