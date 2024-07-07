import os
import json
import orjson
import tqdm
import random
import re
from collections import OrderedDict
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from PIL import Image
import supervision as sv
import cv2
from PIL import Image
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.common.predicate_utils.predicates import load_pddl_yaml, Predicate
from lavis.common.predicate_utils.masks import get_detections, draw_detections, get_detections_h5
from lavis.common.predicate_utils.prompts import get_prompt_function, get_positive_negative_statements



def PIL_load(f, shape=None): 
    im = Image.open(f) 
    if shape is not None: 
        im.draft('RGB',tuple(shape)) #(1008,756)
    return im


class EKVQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, json_dir, split=None, train_samples_portion=None, **kw):
        super().__init__(vis_processor, text_processor, vis_root=vis_root, ann_paths=ann_paths)
        self.annotation_path = ann_paths[0]
        self.split = split or json_dir.split(os.sep)[-1]
        # split_file = self.annotation_path.split('_')[-1].split('.')[0].replace('validation', 'val')

        self.dataset = EpicKitchensDataset(
            self.annotation_path, vis_root, 
            json_dir=json_dir,
            vis_processor=vis_processor, 
            **kw)
        self.annotations = self.dataset.annotations
        self._add_instance_ids()

    @property
    def classes(self):
        return self.dataset.classes

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
        return self.dataset[i]


class VideoFrameDataset(Dataset):
    def __init__(self, 
                 annotations, vis_root, vis_processor, classes, 
                 h5_file=None,
                 qa_prompt=None, 
                 pn_prompt=None,
                 include_detections=False, 
                 filename_format='{video_id}/frame_{i:010d}.jpg', 
                 n_frames=1, 
                 boxes_only=False, 
                 included_object_class_ids=None, 
                 main_object_only=False,
                 prompt_kw=None, 
                 pn_prompt_kw=None,
                 image_load_shape=None, 
                 return_masks=False,
                 n_pos_text=3,
                 n_neg_text=6,
                 lm_context_mode='pre_post',
    ):
        super().__init__()
        self.vis_processor = vis_processor
        self.annotations = annotations
        self.classes = classes
        for i, a in enumerate(annotations):
            a['instance_id'] = i
        self.get_prompt = get_prompt_function(qa_prompt, lm_context_mode)
        self.prompt_kw = {**(prompt_kw or {})}
        self.pn_prompt_kw = {**(pn_prompt_kw or {})}
        self.n_frames = n_frames
        self.vis_root = vis_root
        self.filename_format = filename_format
        self.include_detections = include_detections
        self.boxes_only = boxes_only
        self.included_object_class_ids = included_object_class_ids
        self.main_object_only = main_object_only
        self.image_load_shape = image_load_shape
        self.return_masks = return_masks
        self.n_pos_text = n_pos_text
        self.n_neg_text = n_neg_text
        self.h5_file = None
        if self.include_detections and h5_file is not None:
            print("Using", h5_file)
            self.h5_file = h5py.File(h5_file, 'r', libver='latest')
            idxs = []
            for i, a in enumerate(annotations):
                if a['narration_id'] not in self.h5_file:
                    print(a['narration_id'], 'missing from h5')
                    idxs.append(i)
            self.annotations = [a for i,a in enumerate(annotations) if i not in idxs]
            print(f"dropping samples as they are missing from h5: {len(idxs)}/{len(self.annotations)}")


        # self.im_transform = transforms.ToTensor()
        # self.im_resize = transforms.Resize(size,
        #                                    interpolation=InterpolationMode.BILINEAR,
        #                                    antialias=True)
        # self.mask_resize = transforms.Resize(size,
        #                                      interpolation=InterpolationMode.NEAREST,
        #                                      antialias=True)

    def __len__(self):
        return len(self.annotations)
    
    def load_detections(self, ann):
        raise NotImplementedError()
    
    def __getitem__(self, i, _recursion=0):
        if _recursion > 50: raise RuntimeError("I tried to find some data, I really did... but idk. too many missing.")
        try:
            return self._load_ann(i)
        except Exception as e:
            print("WARNING:", e)
            return self.__getitem__((i + 1)%len(self), _recursion=_recursion+1)

    def _load_ann(self, i):
        ann = self.annotations[i]

        # list frames
        files = self._existing_frames(ann, ann['start_frame'], ann['stop_frame'])
        if not len(files):
            raise RuntimeError(f"no frames for {self._frame_name(video_id=ann['video_id'], i=1)} - {i} {ann['video_id']} {ann['start_frame']} {ann['stop_frame']}")

        # load detection frames
        masks = None
        detections = None
        object_index = None
        if self.include_detections:
            if self.h5_file is None:
                detections = self.load_detections(ann)

            # figure out which frames have detections
            frame_ids = list(files)
            if self.h5_file is not None:
                group = self.h5_file[ann['narration_id']]
                frame_index = group['frame_index'][()]
                has_detection = np.array([i in frame_index for i in frame_ids])
            else:
                has_detection = np.array([i in detections for i in frame_ids])

            # sample frames and get detections
            frames, frame_ids = self._load_frames(files, frame_ids, has_detection * 10 + 1)
            if self.h5_file is not None:
                dets = get_detections_h5(group, frame_ids, frames[0])
            else:
                dets = [get_detections(detections.get(i, {}), x) for x, i in zip(frames, frame_ids)]

            # filter detections
            included_object_class_ids = []
            if self.included_object_class_ids is not None:
                included_object_class_ids.extend(self.included_object_class_ids)
            if self.main_object_only:
                included_object_class_ids.extend(ann['all_noun_classes'])
            if included_object_class_ids:
                dets = [
                    ds[np.isin(ds.class_id, included_object_class_ids)]
                    for ds in dets
                ]
            object_index = list({l for d in dets for l in d.data['labels']})

            # draw detection frames
            if self.return_masks:
                masks = detections.mask
            else:
                det_frames = [
                    draw_detections(x, d, object_index, boxes_only=self.boxes_only) 
                    for x, i, d in zip(frames, frame_ids, dets)
                ]
                # interleave frames
                frames = [x for xs in zip(frames, det_frames) for x in xs]
                # frames = det_frames

        else:
            # load frames
            frames, frame_ids = self._load_frames(files)

        # load question answer
        prompt, target = self.get_prompt(ann, object_index, **self.prompt_kw)
        # load question answer
        all_pos_text, all_neg_text = get_positive_negative_statements(ann, object_index)
        pos_text = np.random.choice(all_pos_text, self.n_pos_text, replace=len(all_pos_text)<self.n_pos_text).tolist()
        neg_text = np.random.choice(all_neg_text, self.n_neg_text, replace=len(all_neg_text)<self.n_neg_text).tolist()

        # get classification vector
        class_targets = class_labels = None
        # if self.classes is not None:
        #     act = ann['action']
        #     states = act.get_state('?a', ann['pre_post'])
        #     states = [s.norm_vars() for s in states]
        #     # class_labels = [str(c) for c in self.classes]
        #     predicates = [Predicate(c).norm_vars() for c in self.classes]
        #     class_targets = torch.as_tensor([1 if c.flip(True) in states else 0 if c.flip(False) in states else -1 for c in predicates])

        video = torch.stack([self.vis_processor(x) for x in frames], dim=0)
        return {
            # **ann,
            'image': video,
            "prompt": prompt,
            "caption": ". ".join(sorted(set(all_pos_text))),
            "text_class": [[x] + neg_text for x in pos_text],
            "text_match": pos_text + neg_text,
            "text_class_targets": torch.zeros(len(pos_text)).long(),
            "text_match_targets": torch.tensor([1]*len(pos_text) + [0]*len(neg_text)).long(),
            "text_input": prompt,
            "text_output": target,
            # metadata
            "narration": ann["narration"],
            "noun": ann["noun"],

            # ID
            "image_id": i,
            "narration_id": ann["narration_id"],
            "instance_id": ann["instance_id"],
            "question_id": ann["instance_id"],
            "sample_id": i,

            "targets": class_targets,
            # "class_labels": class_labels,
            **({'masks': masks} if masks is not None else {})
        }

    
    def _sorted_sample(self, frame_fnames, weights=None):
        if weights is not None:
            weights = np.asarray(weights)
            weights = weights / weights.sum()
        if self.n_frames != 'all':
            frame_fnames = np.random.choice(
                list(frame_fnames), 
                min(self.n_frames or 1, len(frame_fnames)),
                replace=False,
                p=weights)
        return sorted(frame_fnames)

    def _load_frames(self, fs, frame_ids=None, weights=None):
        if frame_ids is None:
            frame_ids = list(fs)
        n = len(fs) if self.n_frames == 'all' else self.n_frames
        n = min(n or 1, len(fs))

        if weights is not None:
            weights = np.asarray(weights)
            weights = weights / weights.sum()

        samples = np.random.choice(frame_ids, len(frame_ids), replace=False, p=weights)
        
        frames = []
        out_frame_ids = []
        for fid in samples:
            for _ in range(3):
                try:
                    frames.append(PIL_load(fs[fid], self.image_load_shape))
                    out_frame_ids.append(fid)
                    break
                except OSError:
                    import time
                    time.sleep(0.1)
            if len(frames) == n:
                break
        
        sort = np.argsort(out_frame_ids)
        out_frame_ids = [out_frame_ids[i] for i in sort]
        frames = [frames[i] for i in sort]
        return frames, out_frame_ids
    
    def _frame_name(self, video_id, i):
        return os.path.join(self.vis_root, self.filename_format.format(video_id=video_id, i=i))

    def _existing_frames(self, ann, start_frame, stop_frame):
        frames = {i: self._frame_name(ann['video_id'], i) for i in range(start_frame, stop_frame+1)}
        frames = {i: f for i, f in frames.items() if os.path.isfile(f)}
        return frames

class EpicKitchensDataset(VideoFrameDataset):
    def __init__(self, annotation_path, vis_root, json_dir, vis_processor, 
                 downsample_count=None, 
                 fake_duplicate_count=None, 
                 outer_buffer=60, 
                 inner_buffer=2, 
                 include_video_ids=None, 
                 exclude_video_ids=None, 
                 filter_verbs=None, 
                 shuffle=True, 
                 predicate_freq_balancing=True, 
                 action_window_mode='pre_post',
                 **kw):
        annotations, predicates, predicate_counts = load_epic_kitchens_dataset(
            annotation_path, 
            downsample_count, 
            outer_buffer, 
            inner_buffer, 
            include_video_ids, 
            exclude_video_ids, 
            filter_verbs, 
            shuffle, 
            predicate_freq_balancing,
            action_window_mode,
        )
        if fake_duplicate_count:
            annotations = annotations * fake_duplicate_count
        self.json_dir = json_dir
        self.classes = [str(p) for p in predicates]
        super().__init__(annotations, vis_root, vis_processor=vis_processor, classes=predicates, **kw)
        self.prompt_kw['predicate_freq'] = predicate_counts
        self.prof_count = 0

    def load_detections(self, ann):
        return load_detections(self.json_dir, ann["narration_id"])




def load_epic_kitchens_dataset(
        annotation_path, 
        downsample_count=None, 
        outer_buffer=60, 
        inner_buffer=2, 
        include_video_ids=None, 
        exclude_video_ids=None, 
        filter_verbs=None, 
        shuffle=True, 
        predicate_freq_balancing=True,
        lm_context_mode='pre_post',
    ):
    df = load_annotation_csv(annotation_path)
    df = load_include_exclude_lists(df, include_video_ids, exclude_video_ids)
    
    if filter_verbs is not None:
        print("filtering", filter_verbs, 'verbs', len(df))
        df = df[df.verb.isin(filter_verbs)]
        print(len(df))

    annotation_dir = os.path.dirname(annotation_path)
    actions, predicates = load_pddl_yaml(f"{annotation_dir}/EPIC_100_conditions.yaml")
    
    if downsample_count:
        downsample_count = int(downsample_count * len(df) if downsample_count <= 1 else downsample_count)
    if shuffle:
        print("SHUFFLING")
        df = df.sample(frac=1, random_state=12345)
    
    annotations = []
    for ann in tqdm.tqdm(df.to_dict('records'), desc=annotation_path.split(os.sep)[-1], leave=False):
        for verb in [ann['verb'], ann.get('verb_norm')]:
            if verb not in actions:
                continue
            action = actions[verb]
            pre = action.get_state(action.vars[0], 'pre')
            post = action.get_state(action.vars[0], 'post')

            duration = ann['stop_frame'] - ann['start_frame']
            inner_buffer = int(min(inner_buffer, duration // 4))
            if lm_context_mode == 'action':
                annotations.append(dict(
                    ann, 
                    start_frame=ann['start_frame'] - outer_buffer,
                    stop_frame=ann['stop_frame'] + outer_buffer,
                    pre_post=None, 
                    action=action,
                    state=post,
                    unknown_state=set(random.sample(predicates, len(predicates))) - set(post) - set(pre),
                ))
                break
            elif lm_context_mode == 'pre_post':
                if pre:
                    annotations.append(dict(
                        ann, 
                        start_frame=ann['start_frame'] - outer_buffer,
                        stop_frame=ann['start_frame'] + inner_buffer,
                        pre_post='pre', 
                        action=action,
                        state=pre,
                        unknown_state=set(random.sample(predicates, len(predicates))) - set(pre),
                    ))
                if post:
                    annotations.append(dict(
                        ann, 
                        start_frame=ann['stop_frame'] - inner_buffer,
                        stop_frame=ann['stop_frame'] + outer_buffer,
                        pre_post='post', 
                        action=action,
                        state=post,
                        unknown_state=set(random.sample(predicates, len(predicates))) - set(post),
                    ))
            if pre or post:
                break
        if downsample_count and len(annotations) >= downsample_count:
            break

    # if 'validation' in annotation_path:
    #     with open('val_annotations.json', 'w') as f:  # debug
    #         json.dump(annotations, indent=2)
    print([d['narration_id'] for d in annotations[:10]])
    print([d['narration_id'] for d in annotations[-10:]])

    predicate_counts = get_predicate_counts(annotations, actions, predicates)
    if not predicate_freq_balancing:
        predicate_counts = None

    return annotations, predicates, predicate_counts


def load_annotation_csv(annotation_path):
    df = pd.read_csv(annotation_path)
    df = df.sort_values(['video_id', 'start_frame'])
    if 'verb' not in df:
        raise RuntimeError(f"verb not in {annotation_path}")

    df = df.dropna(how='any', subset=['start_frame', 'stop_frame', 'verb'])
    df['start_frame'] = df['start_frame'].astype(int)
    df['stop_frame'] = df['stop_frame'].astype(int)

    df['narration_id'] = df['narration_id'].astype(str)
    df['video_id'] = df['video_id'].astype(str)
    
    # # fix errors
    # df.loc[df.narration == 'continue transferring rice from saucepan to plate to plate', 'all_nouns'] = '["rice","saucepan","plate"]'
    # df.loc[df.narration == 'continue transferring rice from saucepan to plate to plate', 'narration'] = 'continue transferring rice from saucepan to plate'
    
    # # convert to timedelta
    # df['start_timestamp'] = pd.to_timedelta(df['start_timestamp'])
    # df['stop_timestamp'] = pd.to_timedelta(df['stop_timestamp'])
    # df['duration'] = (df['stop_timestamp']-df['start_timestamp']).dt.total_seconds()
    
    # parse list strings
    df['all_nouns'] = df.all_nouns.apply(eval)
    if 'all_noun_classes' in df.columns:
        df['all_noun_classes'] = df.all_noun_classes.apply(eval)

    annotation_dir = os.path.dirname(annotation_path)
    try:
        verb_df = load_verbs(annotation_dir)
        df['verb_norm'] = verb_df.key.loc[df.verb_class].values
    except FileNotFoundError:
        df['verb_norm'] = df.verb
    try:
        noun_df = load_nouns(annotation_dir)
        df['noun_norm'] = noun_df.key.loc[df.noun_class].values
    except FileNotFoundError:
        df['noun_norm'] = df.noun

    df['noun'] = df.noun.apply(fix_colon)
    df['noun_norm'] = df.noun_norm.apply(fix_colon)
    df['all_nouns'] = df.all_nouns.apply(lambda xs: [fix_colon(x) for x in xs])
    return df


def load_verbs(annotation_dir):
    verb_df = pd.read_csv(os.path.join(annotation_dir, 'EPIC_100_verb_classes.csv')).set_index('id')
    verb_df['instances'] = verb_df['instances'].apply(eval)
    if 'use' in verb_df.columns: verb_df.loc['use'].instances.append('use-to')
    if 'finish' in verb_df.columns: verb_df.loc['finish'].instances.append('end-of')
    if 'carry' in verb_df.columns: verb_df.loc['carry'].instances.append('bring-into')
    return verb_df


def load_nouns(annotation_dir):
    noun_df = pd.read_csv(os.path.join(annotation_dir, 'EPIC_100_noun_classes_v2.csv')).set_index('id')
    noun_df['instances'] = noun_df['instances'].apply(eval)
    return noun_df


def load_include_exclude_lists(df, include_video_ids=None, exclude_video_ids=None):
    # use video split list
    if exclude_video_ids is not None:
        if isinstance(exclude_video_ids, str):
            exclude_video_ids = pd.read_csv(exclude_video_ids).video_id.tolist()
        print(exclude_video_ids)
        print("excluding", len(exclude_video_ids), 'files', len(df))
        df = df[~df.video_id.isin(exclude_video_ids)]
        print(len(df))
        if not len(df):
            print(df.video_id.unique())
            print(exclude_video_ids)
    elif include_video_ids is not None:
        if isinstance(include_video_ids, str):
            include_video_ids = pd.read_csv(include_video_ids).video_id.tolist()
        print("including", len(include_video_ids), 'files', len(df))
        print(include_video_ids)
        df = df[df.video_id.isin(include_video_ids)]
        if not len(df):
            print(df.video_id.unique())
            print(include_video_ids)
        print(len(df))
    return df


def fix_colon(x):
    xs = x.split(':')
    return ' '.join(xs[1:] + xs[:1])

def get_predicate_counts(annotations, actions, predicates):
    # get predicate count statistics
    predicate_counts = {p.norm_vars(): 0 for p in predicates + [p.flip(False) for p in predicates]}
    for name, act in actions.items():
        for p in act.pre + act.post:
            p = p.norm_vars()
            if p not in predicate_counts:
                print("WARNING:", p, f"is in action {name} but not in the predicate class list")
                predicate_counts[p] = 0

    for ann in annotations:
        for p in ann['state']:
            p = p.norm_vars()
            predicate_counts[p] += 1

    with open('predicate_counts.json', 'w') as f:  # debug
        json.dump({str(k): c for k, c in predicate_counts.items()}, f, indent=2)

    total = sum(predicate_counts.values())
    predicate_counts = {k: total / (c+1) for k, c in predicate_counts.items()}
    return predicate_counts
    

# Masks

def load_detections(annotation_dir, narration_id):  # TODO: slow. pre-dump masks?
    json_path = os.path.join(annotation_dir, f'{narration_id}.json')
    if not os.path.isfile(json_path): 
        print(json_path, "doesn't exist")
        return {}
    # with open(json_path) as f:
    #     data = json.load(f)
    with open(json_path, 'rb') as f:
        data = orjson.loads(f.read())
    return {
        int(d['image']['image_path'].split('/')[-1].split('_')[-1].split('.')[0]): d
        for d in data
    }