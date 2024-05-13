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


@registry.register_task("image_text_pretrain_eval")
class ImageTextPretrainEvalTask(BaseTask):
    def __init__(self):
        super().__init__()

    def valid_step(self, model, samples):
        itc_targets = samples['text_class_targets']  # [batch, sample]
        itm_targets = samples['text_match_targets']  # [batch, sample]
        itc_pred = model.itc(samples)  # [batch, sample, class_proba]
        itm_pred = model.itm(samples)  # [batch, sample, 2]
        caption_pred = model.generate(samples)

        results = []
        with open(f'{registry.get_path("result_dir")}/result_stream.txt', 'a') as fh:
            n = len(samples["image_id"])
            for i in range(n):
                r = {
                    "text_class": samples["text_class"][i],  # str[n_pos, n_neg+1]
                    "text_match": samples["text_match"][i],  # str[n_pos + n_neg]
                    "text_class_true": samples['text_class_targets'][i].cpu().tolist(), # [n_pos]
                    "text_match_true": samples["text_match_targets"][i].cpu().tolist(), # [n_pos + n_neg]
                    "text_class_pred": itc_pred[i].cpu().tolist(), # [n_pos, n_neg+1]
                    "text_match_pred": itm_pred[i].cpu().tolist(), # [n_pos + n_neg, 2]
                    "caption_true": samples["caption"][i],  # str
                    "caption_pred": caption_pred[i],  # str

                    "image_id": samples["image_id"][i], 
                    "narration_id": samples["narration_id"][i], 
                    "noun": samples["noun"][i], 
                }
                results.append(r)
                fh.write(f'{json.dumps(r)}\n')
                if samples['sample_id'][i] in self.sample_index:
                    print(samples["narration"][i])
                    print("Match:")
                    print(samples["text_match"][i])
                    print(samples["text_match_targets"][i].cpu().tolist())
                    print(np.round(itm_pred[i][:, 1].double().cpu().numpy(), 2).tolist())
                    print("Class:", samples["text_class"][i][0])
                    for j in range(len(samples["text_class"][i])):
                        print(samples["text_class"][i][j][0])
                        print(samples["text_class_targets"][i][j].cpu().tolist(), np.round(itc_pred[i][j].double().cpu().numpy(), 2).tolist())

                # if samples['sample_id'][i] in self.sample_index:
                #     print("Sample:", samples['sample_id'][i], samples["narration_id"][i], samples["narration"][i])
                #     print("in:", samples['text_input'][i])
                #     print("pred:", answer_pred[i])
                #     print("true:", samples["text_output"][i])
                    self.result_table.add_data(
                        samples['sample_id'][i],
                        wandb.Video(norm_video(samples["image"][i]).cpu().numpy(), fps=3) 
                        if samples["image"].ndim == 5 else
                        wandb.Image(samples["image"][i].cpu()),

                        samples["caption"][i] + '\n' + caption_pred[i], 
                        '\n'.join(
                            f'{t} {i} {p:.0%}' 
                            for t, i, p in zip(samples["text_match"][i], samples["text_match_targets"][i].tolist(), itm_pred[i, :, 1].double().cpu().tolist())),
                        ' | '.join(samples["text_class"][i][0]) + '\n' + '\n'.join(
                            f'{t[0]} {i} {np.round(p, 2).tolist()}' 
                            for t, i, p in zip(samples["text_class"][i], samples["text_class_targets"][i].tolist(), itc_pred[i].double().cpu().numpy())
                        ),

                        samples["narration_id"][i],  # Narration ID
                        samples["narration"][i]  # Narration text
                    )
        return results


    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        results = json.load(open(result_file, "r"))

        # calculate metrics
        errors = []
        acc = []
        for res in results:
            try:
                acc.append(jaccard_score(res["caption_pred"], res["caption_true"]))
            except Exception as e:
                print("Could not parse", res["caption_pred"], e)
                errors.append(res["caption_pred"])
                acc.append(0)
        
        y_true_flat = np.array([x for d in results for x in d['text_class_true']]).astype(int)
        y_pred_flat = np.array([x for d in results for x in d['text_class_pred']]).astype(float)
        y_pred_flat = np.argmax(y_pred_flat, 1)
        class_accuracy = accuracy_score(y_true_flat, y_pred_flat)
        # class_precision = precision_score(y_true_flat, y_pred_flat)
        # class_recall = recall_score(y_true_flat, y_pred_flat)
        # class_f1 = f1_score(y_true_flat, y_pred_flat)

        y_true_flat = np.array([x for d in results for x in d['text_match_true']]).astype(int)
        y_pred_flat = np.array([x for d in results for x in d['text_match_pred']]).astype(float)
        y_pred_flat = np.argmax(y_pred_flat, 1)
        match_accuracy = accuracy_score(y_true_flat, y_pred_flat)
        # match_precision = precision_score(y_true_flat, y_pred_flat)
        # match_recall = recall_score(y_true_flat, y_pred_flat)
        # match_f1 = f1_score(y_true_flat, y_pred_flat)

        # report metrics
        lm_accuracy = np.mean(acc)
        metrics = {
            "agg_metrics": class_accuracy, 
            "lm_accuracy": lm_accuracy,
            "class_accuracy": class_accuracy, 
            # "class_precision": class_precision,
            # "class_recall": class_recall,
            # "class_f1": class_f1,
            "match_accuracy": match_accuracy, 
            # "match_precision": match_precision,
            # "match_recall": match_recall,
            # "match_f1": match_f1,
            "split": split,
        }

        with open(os.path.join(registry.get_path("output_dir"), f"log.txt"), "a") as f:
            f.write(json.dumps(metrics) + "\n")
        with open(os.path.join(registry.get_path("output_dir"), f"parse_error.txt"), "a") as f:
            for x in errors:
                f.write(f'{x}\n')

        logging.info(metrics)

        return metrics
        
    
    def before_evaluation(self, model, dataset, **kwargs):
        print("before eval",kwargs)
        super().before_evaluation(model, dataset, **kwargs)
        self.sample_index = np.random.choice(len(dataset), min(60, len(dataset)), replace=False)
        print("eval samples:", self.sample_index, len(dataset))
        self.result_table = wandb.Table(columns=["index", "image", "caption", "text_match", "text_class", "narration_id", "narration"])

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



def norm_video(img):
    img = img.float()
    for t in img:
        low, high = float(t.min()), float(t.max())
        t.clamp_(min=low, max=high)
        t.sub_(low).div_(max(high - low, 1e-5))
    img = img.mul(255).clamp(0, 255).byte()
    return img



CLEAN = VQACleaner()


def split_text(text, *delims, n=-1):
    for d in delims:
        if d in text:
            return [y.strip() for y in text.split(d, n) if y.strip()]
    return ([text] + ['']*n) if n else [text]


def jaccard_score(y_pred, y_true):
    # XXX: repetition/contradiction
    y_pred = {CLEAN(x) for x in split_text(y_pred, '.', ',', n=-1)}
    y_true = {CLEAN(x) for x in split_text(y_true, '.', ',', n=-1)}
    return len(y_pred & y_true) / (len(y_pred | y_true) or 1)
