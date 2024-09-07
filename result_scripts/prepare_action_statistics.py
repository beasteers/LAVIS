# python -m torch.distributed.run --nproc_per_node=1 --master_port=25676 train.py --cfg-path lavis/projects/instructblip/train/epic-kitchens/finetune_instructblip_epic_kitchens_4_eval_ssv2.yaml   --name ekos_ssv2 --tags cross --options run.resume_ckpt_path=output/results/epic_kitchens/epic_kitchens_4/20240307072/checkpoint_best.pth
# python -m torch.distributed.run --nproc_per_node=1 --master_port=25676 train.py --cfg-path lavis/projects/instructblip/train/epic-kitchens/finetune_instructblip_epic_kitchens_4_eval_ssv2.yaml   --name ekos_ssv2 --tags cross --options run.resume_ckpt_path=output/results/epic_kitchens/epic_kitchens_4/20240307072/checkpoint_best.pth
import os
import sys
sys.path.append('.')
import glob
import json
from collections import defaultdict
import functools
from concurrent.futures import ProcessPoolExecutor
# import mqdm
import tqdm
tqdm.tqdm.pandas()
import orjson
import h5py
import cv2
from PIL import Image
import numpy as np
from numpy.polynomial import Polynomial
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix, precision_score, recall_score
import shapely
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.ndimage import gaussian_filter1d
import torch
from lavis.common.predicate_utils.predicates import load_pddl_yaml
from lavis.common.predicate_utils.masks import get_detections, draw_detections, get_detections_h5
# from lavis.datasets.ekos_datasets import load_annotation_csv, load_nouns, load_verbs

HAND_IDS = [11,300,301,303,304]
VISOR_SCALE = np.array([854, 480])  # x, y
DIR = '/rgb_frames'

# ---------------------------------------------------------------------------- #
#                               Load annotations                               #
# ---------------------------------------------------------------------------- #

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


def fix_colon(x):
    xs = x.split(':')
    return ' '.join(xs[1:] + xs[:1])


# ---------------------------------------------------------------------------- #
#                                 Compute stats                                #
# ---------------------------------------------------------------------------- #

def process_video_actions(df):
    st = df['start_frame'].values
    et = df['stop_frame'].values
    END=et.max()
    nearest_before = np.array([et[et < s].max(initial=0) for s in st])
    nearest_after = np.array([st[st > e].min(initial=END) for e in et])
    overlap = (
        (st[:, None] <= et[None]) & 
        (et[:, None] >= st[None])
    ).sum(1) - 1

    df = df[['narration_id']].copy()
    df['pre_frame_gap'] = st - nearest_before
    df['post_frame_gap'] = nearest_after - et
    df['overlapping_actions'] = overlap
    return df


def agg_series(key, xs, include=()):
    xs = np.array(xs)
    xs = xs[~np.isnan(xs)]
    if not len(xs):
        return {}
    stats = {
        'min': np.min(xs),
        'max': np.max(xs),
        'mean': np.mean(xs),
        'median': np.median(xs),
    }
    if 'total' in include:
        stats['total'] = np.sum(xs)
    return {f'{key}_{k}': v for k, v in stats.items()}

def compute_centroid(contours):
    M_total = {'m00': 0, 'm10': 0, 'm01': 0}
    for contour in contours:
        M = cv2.moments(contour)
        M_total['m00'] += M['m00']
        M_total['m10'] += M['m10']
        M_total['m01'] += M['m01']

    # Calculate the centroid
    return (
        int(M_total['m10'] / M_total['m00']) if M_total['m00'] else np.nan,
        int(M_total['m01'] / M_total['m00']) if M_total['m00'] else np.nan,
    )

def compute_iou(poly1, poly2):
    if not poly1 or not poly2:
        return np.nan
    # Create polygon masks
    img1 = np.zeros(VISOR_SCALE[::-1], dtype=np.uint8)
    img2 = np.zeros(VISOR_SCALE[::-1], dtype=np.uint8)
    cv2.fillPoly(img1, poly1, 1)
    cv2.fillPoly(img2, poly2, 1)

    # Calculate intersection over union
    intersection = np.logical_and(img1, img2)
    union = np.logical_or(img1, img2)
    iou = np.sum(intersection) / max(np.sum(union), 1e-6)
    return iou

def compute_frame_to_frame_iou_consistency(polygons_per_frame):
    return [
        compute_iou(polygons_per_frame[i-1], polygons_per_frame[i])
        for i in range(1, len(polygons_per_frame))
    ]

def compute_poly_disappearances(polygons_per_frame):
    return [
        compute_iou(polygons_per_frame[i-1], polygons_per_frame[i])
        for i in range(1, len(polygons_per_frame))
    ]

def compute_aspect_ratio(polygons):
    all_points = np.concatenate(polygons, axis=0)
    x, y, w, h = cv2.boundingRect(np.array(all_points, dtype=np.int32))
    aspect_ratio = w / h if h != 0 else np.nan
    return aspect_ratio

def load_segments(segment_str_array):
    # segments = [json.loads(s.decode()) for s in segment_str_array]
    # xs = np.array(xs)#(1, 1, 102, 1, 2)
    # if len(xs) and (xs.shape[1]!=1 or xs.shape[3] != 1):
    #     1/0
    return [
        np.array(si, dtype=np.int32) 
        for s in segment_str_array 
        for si in json.loads(s.decode())
    ]


def get_narration_objects(narration_id, noun_class, all_noun_classes, start_frame, stop_frame, h5f, buffer_size=30):
    g = h5f[narration_id]
    # names = g['names']
    sources = g['source'][()].astype(str)
    model_sources = g['model_source'][()].astype(str)
    segments = g['segments'][()]
    class_ids = g['class_ids'][()]
    frame_index = g['frame_index'][()]
    assert not (set(sources) - {'visor', 'egohos', 'xmem'}), f"Wait hold up. check: {set(sources) - {'visor', 'egohos', 'xmem'}}"

    stats = {
        'narration_id': narration_id,
        'action_duration_frames': stop_frame - start_frame,
        'number_total_objects': len(np.unique(class_ids)),
    }
    for key, center_frame in {'pre': start_frame, 'post': stop_frame}.items():
        idx = frame_index[(frame_index >= center_frame - buffer_size) & (frame_index < center_frame + buffer_size)]
        frames = np.sort(np.unique(idx))
        # frames = np.arange(center_frame - buffer_size, center_frame + buffer_size + 1)
        current_frame = np.isin(frame_index, idx)
        active_noun = np.isin(class_ids, [noun_class])
        is_hand = np.isin(class_ids, HAND_IDS)
        is_bg = ~np.isin(class_ids, [noun_class] + HAND_IDS)

        # get segments of the main noun for each frame between start and end
        # list(frames) of list(polygons) of array(N points, 1, 2)
        active_polys = [load_segments(segments[(frame_index == i) & active_noun]) for i in frames]
        hand_polys = [load_segments(segments[(frame_index == i) & is_hand]) for i in frames]
        bg_polys = [load_segments(segments[(frame_index == i) & is_bg]) for i in frames]

        active_mpolys = [shapely.MultiPolygon([shapely.Polygon(p[:, 0]) for p in mp]) for mp in active_polys]
        hand_mpolys = [shapely.MultiPolygon([shapely.Polygon(p[:, 0]) for p in mp]) for mp in hand_polys]
        bg_mpolys = [shapely.MultiPolygon([shapely.Polygon(p[:, 0]) for p in mp]) for mp in bg_polys]

        # polygon distance
        hand_distance = [p1.distance(p2) for p1, p2 in zip(active_mpolys, hand_mpolys)]
        bg_distance = [p1.distance(p2) for p1, p2 in zip(active_mpolys, bg_mpolys)]

        # polygon counts
        active_counts = [len(x) for x in active_polys]
        bg_counts = [len(x) for x in bg_polys]
        active_visible = [len(x)>0 for x in active_polys]
        hand_visible = [len(x)>0 for x in hand_polys]
        bg_visible = [len(x)>0 for x in bg_polys]
        active_visible_changes = np.abs(np.diff(active_visible))
        hand_visible_changes = np.abs(np.diff(hand_visible))
        bg_visible_changes = np.abs(np.diff(bg_visible))
        
        # polygon centroid
        centroids = np.array([compute_centroid(c) for c in active_polys if len(c)])
        if len(centroids):
            centroids = centroids[~np.isnan(centroids).any(1)]
        
        # movement of the centroid
        displacement = np.diff(centroids, axis=0)
        displacement_abs = np.linalg.norm(displacement, axis=-1) if len(displacement) else displacement

        # polygon area
        area = np.array([sum(cv2.contourArea(xi) for xi in x) for x in active_polys if len(x)]) / np.prod(VISOR_SCALE)
        area = area[area!=0]
        log_area = np.log(area + 1)
        sqrt_area = area ** 0.5
        delta_area = np.diff(area, axis=0)
        delta_log_area = np.diff(log_area, axis=0)
        delta_sqrt_area = np.diff(sqrt_area, axis=0)
        delta_log_area_norm = delta_log_area / max(area, default=1)
        aspect_ratio = np.array([compute_aspect_ratio(c) for c in active_polys if len(c)])
        log_aspect_ratio = np.log(aspect_ratio+1)
        delta_aspect_ratio = np.diff(aspect_ratio, axis=0)
        delta_log_aspect_ratio = np.diff(log_aspect_ratio, axis=0)
        # delta_aspect_ratio_norm = delta_aspect_ratio / max(aspect_ratio, default=1)
        iou_consistency = compute_frame_to_frame_iou_consistency(active_polys)
        
        # how close is the polygon to the borders of the image?
        boundary = [(2*(np.concatenate(c)[:, 0] / VISOR_SCALE - 0.5)) for c in active_polys if len(c)]
        boundary_proximity = [np.abs(x).max() for x in boundary]
        x_boundary = [np.abs(x[:, 0]).max() for x in boundary]
        y_boundary = [(x[:, 1]).max() for x in boundary]
        top_boundary = [(x[:, 1]).min() for x in boundary]

        # how visible is the object?
        # - x visible (number of active objects)
        # - size
        #   - x area, log area, sqrt area, aspect ratio
        # - occlusion
        #   - x change in shape/aspect ratio
        #   - x proximity to other background objects, hands
        # - blurry
        # - low-light
        # - camera movement
        # - x out of frame - bottom, top, left/right proximity

        # how much distraction is there?
        # - x number of background objects
        # - x number of hands

        # how trustworthy is the polygon?
        # - x annotation source
        # - x annotation original model source (before xmem)
        # - x iou consistency over time

        # what is happening to the object?
        # - x speed, displacement, total distance

        # how ambiguous is the question?
        # are there multiple objects of the same class?
        # - x number of polygons

        # what other things are happening?
        # - x action duration
        # - x gap between neighboring actions
        # - x overlapping actions

        # time series aggregates: min/max/mean/median
        # x source
        # x number of polygons
        # x touching sides
        # x action duration
        # x area of polygon, bounding box
        # x displacement of centroid
        # x distance traveled
        stats.update({
            **({
                f'{key}_source_percent_{stype}_{s}': (srcs == s).mean()
                for stype, srcs in {
                    'active': sources[current_frame & active_noun], 
                    'hand': sources[current_frame & is_hand],
                    'obj': sources[current_frame], 
                    'model_active': model_sources[current_frame & active_noun], 
                    'model_hand': model_sources[current_frame & is_hand],
                    'model_obj': model_sources[current_frame], 
                }.items()
                for s in {'visor', 'egohos', 'xmem'} | set(srcs)
            }),
            
            f'{key}_number_objects_total': len(np.unique(class_ids[np.isin(frame_index, idx)])),
            f'{key}_number_objects_active': len(np.unique(class_ids[current_frame & active_noun])),
            f'{key}_number_objects_bg': len(np.unique(class_ids[current_frame & is_bg])),
            f'{key}_number_objects_hands': len(np.unique(class_ids[current_frame & is_hand])),
            **agg_series(f'{key}_number_polygons_active', active_counts),
            **agg_series(f'{key}_number_polygons_bg', bg_counts),
            **agg_series(f'{key}_is_visible_active', active_visible),
            **agg_series(f'{key}_is_visible_hand', hand_visible),
            **agg_series(f'{key}_is_visible_bg', bg_visible),
            **agg_series(f'{key}_visible_changes_active', active_visible_changes, ['total']),
            **agg_series(f'{key}_visible_changes_hand', hand_visible_changes, ['total']),
            **agg_series(f'{key}_visible_changes_bg', bg_visible_changes, ['total']),
            # poly distance
            **agg_series(f'{key}_polygon_distance_to_hand', hand_distance),
            **agg_series(f'{key}_polygon_distance_to_bgobjects', bg_distance),
            # area
            **agg_series(f'{key}_area', area),
            **agg_series(f'{key}_area_log', log_area),
            **agg_series(f'{key}_area_sqrt', sqrt_area),
            **agg_series(f'{key}_area_delta', delta_area),
            **agg_series(f'{key}_area_delta_log', delta_log_area),
            **agg_series(f'{key}_area_delta_sqrt', delta_sqrt_area),
            **agg_series(f'{key}_area_delta_log_norm', delta_log_area_norm),
            **agg_series(f'{key}_aspect_ratio', aspect_ratio),
            **agg_series(f'{key}_aspect_ratio_log', log_aspect_ratio),
            **agg_series(f'{key}_aspect_ratio_delta', delta_aspect_ratio),
            **agg_series(f'{key}_aspect_ratio_delta_log', delta_log_aspect_ratio),
            **agg_series(f'{key}_iou_consistency', iou_consistency),
            # displacement
            **agg_series(f'{key}_centroid_displacement', displacement_abs, ['total']),
            **agg_series(f'{key}_boundary_proximity', boundary_proximity),
            **agg_series(f'{key}_boundary_proximity_x', x_boundary),
            **agg_series(f'{key}_boundary_proximity_bottom', y_boundary),
            **agg_series(f'{key}_boundary_proximity_top', top_boundary),
            
        })
    return stats


def get_stats(annotation_dir='/scratch/bs3639/ego2023/epic-kitchens-100-annotations', h5_path='/scratch/bs3639/EKOS_val_3.h5', out_path='stats.csv'):
    if os.path.isfile(out_path):
        df = pd.read_csv(out_path)
        df = df.set_index('narration_id').sort_index()
    else:
        df = load_annotation_csv(f'{annotation_dir}/EPIC_100_validation.csv')
        act_df = df.groupby('video_id').apply(process_video_actions).set_index('narration_id')

        h5_file = h5py.File(h5_path, 'r', libver='latest')
        ann_df = pd.DataFrame([
            get_narration_objects(r.narration_id, r.noun_class, r.all_noun_classes, r.start_frame, r.stop_frame, h5_file)
            for i, r in tqdm.tqdm(df.iterrows(), total=len(df))
        ]).set_index('narration_id')

        ddf = df.set_index('narration_id')
        ddf = pd.concat([ddf, act_df, ann_df], axis=1)
        df = ddf.sort_index()
        df.to_csv(out_path)

    df['log_action_duration_frames'] = np.log(df['action_duration_frames'])
    return df


def get_changes_for_video(vdf, actions):
    video_id = vdf.name
    last_frame = vdf.stop_frame.max()
    narration_ids = vdf.narration_id.unique()
    full_index = [(n, k) for n in narration_ids for k in ['pre', 'post']]
    
    dfs = []
    for (noun, noun_class), nvdf in vdf.groupby(['noun_norm', 'noun_class']):
        changes = []
        predicates = set()
        for _, row in nvdf.iterrows(): #tqdm.tqdm(, total=len(nvdf), desc=f'{vdf.name} {noun}'):
            action = next((
                actions[v] for v in [row.verb, row.verb_norm]
                if v in actions
            ), None)
            if action is None:
                tqdm.tqdm.write(f"skipping {row.verb} {row.verb_norm}")
                continue

            assert '?a' in action.vars, f"?a not in {action.vars}"
            preconditions = action.get_state("?a", "pre")
            postconditions = action.get_state("?a", "post")
            predicates.update(p.name for p in preconditions)
            predicates.update(p.name for p in postconditions)
            # tqdm.tqdm.write(f'{noun} {row.verb}')
            # tqdm.tqdm.write(f'pre: {preconditions}')
            # tqdm.tqdm.write(f'post: {postconditions}')
            changes.append({
                'video_id': video_id,
                'noun': noun,
                'noun_class': noun_class,
                'verb': row.verb,
                'verb_class': row.verb_class,
                'action_verb': action.name,
                'action_stage': 'pre',
                'narration_id': row.narration_id,
                'frame_index': row.start_frame,
                **({
                    p.name: p.is_true()
                    for p in preconditions
                }),
            })
            changes.append({
                'video_id': video_id,
                'noun': noun,
                'action_stage': 'post',
                'narration_id': row.narration_id,
                'frame_index': row.stop_frame,
                **({
                    p.name: p.is_true()
                    for p in postconditions
                }),
            })
        if not len(changes):
            tqdm.tqdm.write(f"skipping {video_id} {noun} - no changes recorded")
            continue
        cols = list(predicates)
        changes_df = pd.DataFrame(changes)
        # print(changes_df[cols].mean())
        # print(pd.isna(changes_df[cols]).mean())
        with pd.option_context("future.no_silent_downcasting", True):
            changes_df = changes_df.ffill().bfill().infer_objects(copy=False)
            # print(changes_df[cols].mean())
            # print(pd.isna(changes_df[cols]).mean())
            vis_df = changes_df.groupby('frame_index', group_keys=True).last()
            vis_df = vis_df.reindex(np.arange(last_frame+1)).ffill().bfill().infer_objects(copy=False)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(vis_df[cols].values.T.astype(int), aspect='auto', vmin=0, vmax=1, cmap='bone', interpolation='nearest')
        plt.yticks(range(len(cols)), cols)
        frames = vis_df.index[np.linspace(0, vis_df.index.max(), 16).astype(int)]
        plt.xticks(frames, np.round((frames / 30) / 60, decimals=1))
        plt.axvline(changes_df.frame_index.min(), linestyle='dotted', color='red')
        plt.axvline(changes_df.frame_index.max(), linestyle='dotted', color='red')
        plt.xlabel("Minutes")
        plt.title(f'{video_id} {noun}')
        plt.colorbar()
        os.makedirs(f'plots_changes/{video_id}', exist_ok=True)
        plt.savefig(f'plots_changes/{video_id}/{noun}.png')
        changes_df.to_csv(f'plots_changes/{video_id}/_{noun}.csv')
        # print(f'plots_changes/{video_id}/{noun}.png')
        # input()

        dfs.append(changes_df)
    changes_df = pd.concat(dfs)
    return changes_df


def expand_narrations_across_nouns(out_path='expanded_changes.csv'):
    if os.path.isfile(out_path):
        return pd.read_csv(out_path)
    print("Expanding state change index for nouns across all narrations in a video")
    changes_df = pd.concat([pd.read_csv(f) for f in glob.glob(f'plots_changes/*/_*.csv')])
    dfs = []
    for video_id, cdf in changes_df.groupby('video_id'):
        idx_cols = ['video_id', 'narration_id', 'action_stage', 'frame_index']
        index = cdf[idx_cols].value_counts().index
        for noun, ndf in cdf.groupby('noun'):
            ndf2 = ndf.set_index(idx_cols).reindex(index).reset_index()
            # x = ndf2.isna().mean()
            # x=x[(x<1)&(x>0)]
            # print(len(ndf), len(ndf2), x if len(x) else None)
            with pd.option_context("future.no_silent_downcasting", True):
                ndf2 = ndf2.sort_values('frame_index').ffill().bfill().infer_objects(copy=False)
            # x = ndf2.isna().mean()
            # # x=x[(x<1)&(x>0)]
            # print(noun, len(ndf), len(ndf2), x if len(x) else None)
            # input()
            dfs.append(ndf2)

    changes_df = pd.concat(dfs)
    # print(changes_df.columns)
    # if input('>?'):from IPython import embed;embed()
    changes_df.to_csv(out_path)
    return changes_df


def calculate_windowed_switches(changes_df, out_path='window_changes.csv', out_dir='output_window_changes', window_sizes=np.array([1, 15, 30, 60, 90, 120, 180, 300, 600, 900, 1200]), fps=30):
    if os.path.isfile(out_path):
        return pd.read_csv(out_path)
    window_sizes = np.sort(np.array(window_sizes) * fps)[::-1]
    dfs = []
    boolean_columns = [
        c for c in changes_df.columns
        if not (set(changes_df[c].dropna().unique()) - {0, 1})
    ]
    # print(boolean_columns)
    print("Computing state switch counts over windows", window_sizes)
    changes_df = changes_df.reset_index().set_index('frame_index')
    for video_id, vdf in tqdm.tqdm(changes_df.groupby('video_id')):
        vid_rows = []
        for noun, ndf in tqdm.tqdm(vdf.groupby('noun'), desc=video_id):
            chdf = ndf[boolean_columns].dropna(how='all', axis=1)
            chdf = chdf != chdf.shift()
            for frame_index, row in tqdm.tqdm(ndf.iterrows(), total=len(ndf), desc=noun, leave=False):
                win_df = chdf
                for window_size in window_sizes:
                    win_df = win_df.loc[frame_index - window_size:frame_index + window_size]
                    # print(win_df)
                    vid_rows.append({
                        'video_id': video_id,
                        'noun': noun,
                        'narration_id': row.narration_id,
                        'action_stage': row.action_stage,
                        'window_size': window_size,
                        'frame_index': frame_index,
                        'start_frame_index': frame_index - window_size,
                        'stop_frame_index': frame_index + window_size,
                        **win_df.sum().to_dict(),
                    })

        dfs.extend(vid_rows)
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(vid_rows).to_csv(f'{out_dir}/{video_id}.csv')
    df = pd.DataFrame(dfs)
    df.to_csv(out_path)
    if input('>?'):from IPython import embed;embed()
    return df   

# window changes
#  - 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@functools.lru_cache(1)
def load_model():
    import torch
    from PIL import Image
    from lavis.models import load_model_and_preprocess
    from lavis.processors import load_processor
    torch.no_grad().__enter__()
    model, vis_processors, text_processors = load_model_and_preprocess("blip2_qformer2", "pretrain", device=device, is_eval=True)
    
    ck_path = 'lavis/output/results/s20bn/s20bn_pretrain/20240513160/checkpoint_best.pth'
    # ck_path = 'lavis/output/results/epic_kitchens/epic_kitchens_pretrain/20240513160/checkpoint_best.pth'
    checkpoint = torch.load(ck_path)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.vis_processors = vis_processors
    model.text_processors = text_processors
    return model



@functools.lru_cache(1)
def load_hdf5(path):
    return h5py.File(path, 'r', libver='latest')


def get_detection_images(video_id, narration_id, frame_index, noun_classes, h5_path, mask=None):
    img = Image.open(f'{DIR}/{video_id}/frame_{frame_index:010d}.jpg')
    h5_file = load_hdf5(h5_path) if isinstance(h5_path, str) else h5_path
    det_imgs = []
    ds = get_detections_h5(h5_file[narration_id], [frame_index], img, mask=mask)
    for noun_class in noun_classes:
        ds_n = [d for d in ds if d['class_id'] == noun_class]
        det_imgs.append(draw_detections(img, ds_n[0], ds_n[0].data['labels'].tolist()) if len(ds_n) else img)
    return det_imgs

def get_detection_image(video_id, narration_id, frame_index, noun_class, h5_path, mask=None):
    img = Image.open(f'{DIR}/{video_id}/frame_{frame_index:010d}.jpg')
    h5_file = load_hdf5(h5_path) if isinstance(h5_path, str) else h5_path
    det_imgs = []
    ds = get_detections_h5(h5_file[narration_id], [frame_index], img, noun_classes=[noun_class], mask=mask)
    return draw_detections(img, ds[0], ds[0].data['labels'].tolist()) if len(ds) else img


def predict_itm(model, imgs, dets, questions):
    Xs = []
    for img, det_img in zip(imgs, dets):
        Xs.append(torch.stack([
            img, #model.vis_processors["eval"](img), 
            det_img, #model.vis_processors["eval"](det_img),
        ]))
    Ximg = torch.stack(Xs).to(device)
    Xtext = [[model.text_processors["eval"](q) for q in qs] for qs in questions]

    # itm_pred = model.itm({"image": Ximg, "text_match": Xtext})
    itm_pred, index = model.itm_flat({"image": Ximg, "text_match": Xtext})
    itm_pred = itm_pred.cpu().numpy()
    itm_pred = [
        itm_pred[index == i][:, 1]
        for i in range(len(questions))
    ]
    return itm_pred


# class MetaDict(defaultdict):
#     def __init__(self, meta, **kw):
#         super().__init__(lambda: [])
#         self.meta = dict(meta, **kw)

#     def validate_length(self):
#         lens = set(len(v) for v in self.values())
#         assert len(lens) == 1
#         return list(lens)[0]

#     def entries(self):
#         keys = set(self)
#         n = self.validate_length()
#         return [
#             dict(self.meta, **({k: self[k][i] for k in keys}))
#             for i in range(n)
#         ]



def predict_video(model, ann_df, changes_df, predicates, h5_file, h5_path, noun_df, output_dir='output_video_predictions_ssv2', buffer=30, n_boundary_frames=5, n_inter_frames=3, pbar=None):
    '''
    for each narration
        get nouns from detections
        for each noun
            which frames is it visible?
            sample 5 frames where it is visible
            get questions from changes_df
            pred = model([frame, det(frame, noun)], questions)
    '''
    if model is None:
        model = load_model()
    if h5_file is None:
        h5_file = load_hdf5(h5_path)
    os.makedirs(output_dir, exist_ok=True)
    # out_path = f'{output_dir}/pred.csv'
    ann_df = ann_df.reset_index().set_index('narration_id')
    noun_df2 = ann_df.groupby('noun_class').noun_norm.last().sort_index()
    changes_df = changes_df.reset_index().set_index(['narration_id', 'action_stage', 'noun'])
    changes_narration_ids = changes_df.index.get_level_values(0)

    predicate_dict = {p.name: p.norm() for p in predicates}
    predicate_cols = list(set(predicate_dict) & set(changes_df.columns))

    predictions = []
    # try:
    with ProcessPoolExecutor(max_workers=22) as pool:
        # loop over videos (just for prettier progress)
        for video_id, vdf in tqdm.tqdm(ann_df.groupby('video_id')):

            vid_csv_path = f'{output_dir}/{video_id}.csv'
            if os.path.isfile(vid_csv_path):
                try:
                    predictions.extend(pd.read_csv(vid_csv_path).to_records())
                    continue
                except Exception:
                    pass

            cdf_ = changes_df[changes_df.video_id == video_id]
            changes_nouns = cdf_.index.get_level_values(-1).unique()
            changes_narration_ids = cdf_.index.get_level_values(0).unique()

            # loop over narrations
            video_predictions = []
            for narration_id, row in tqdm.tqdm(vdf.iterrows(), total=len(vdf), desc=video_id):
                if narration_id not in changes_narration_ids:
                    tqdm.tqdm.write(f"\n\n\n!!! {narration_id} {row.verb} {row.narration} not in changes_df !!! \n\n\n")
                    continue

                # load ekos detections
                g = h5_file[narration_id]
                # nouns = g['names'][()].astype(str)
                class_ids = g['class_ids'][()]
                frame_index = g['frame_index'][()]

                known_nouns = np.isin(noun_df2.reindex(class_ids), changes_nouns)
                frame_weights = pd.Series(frame_index[known_nouns]).value_counts()

                cdf = changes_df.loc[narration_id]
                start_frame = row.start_frame
                stop_frame = row.stop_frame

                # loop over narration pre/post segments
                for action_stage, (center_frame, frame0, frame1, n_frames) in {
                    'pre': (start_frame, start_frame - buffer, start_frame + buffer, n_boundary_frames), 
                    'mid': (start_frame, start_frame, stop_frame, n_inter_frames), 
                    'post': (stop_frame, stop_frame - buffer, stop_frame + buffer, n_boundary_frames),
                }.items():
                    if frame1 <= frame0:
                        continue

                    frame_slice = (frame_index >= frame0 if frame0 is not None else True) & (frame_index < frame1 if frame1 is not None else True)
                    current_frame = frame_slice & known_nouns

                    visible_index = frame_index[current_frame]
                    if not len(visible_index):
                        tqdm.tqdm.write(f"{narration_id} {row.verb} {row.narration} no usable state changes {set(noun_df.key[class_ids[frame_slice]])}")
                        continue
                    weight = np.array([frame_weights.get(f, 0) for f in visible_index]) + 1e-3
                    weight = weight / weight.sum()
                    visible_index = np.random.choice(visible_index, min(n_frames, len(visible_index)), p=weight)
                    unique_noun_classes = np.unique(class_ids[current_frame])

                    imgs_ = [
                        Image.open(f'{DIR}/{video_id}/frame_{f_idx:010d}.jpg')
                        for f_idx in visible_index
                    ]
                    imgs_ = [model.vis_processors["eval"](img) for img in imgs_]
                    imgs = []
                    futs = []
                    nouns = []
                    noun_classes = []
                    pred_index = []
                    questions = []
                    states = []
                    ys_true = []
                    for i, f_idx in enumerate(visible_index):
                        for noun_class in unique_noun_classes:
                            noun = noun_df2.loc[noun_class]
                            if action_stage == 'mid':
                                noun_state = cdf.loc[('pre', noun), predicate_cols].dropna()
                                noun_state[:] = np.nan
                            else:
                                noun_state = cdf.loc[(action_stage, noun), predicate_cols].dropna()
                            # generate the model questions (and gt)
                            qs_ = [predicate_dict[c].format(noun, default='something') for c in noun_state.index]
                            qs_ = [f'is {q}' if not q.startswith('can ') else q for q in qs_]
                            qs_ = [f'{noun} {q}' for q in qs_]
                            
                            states.append(noun_state.index)
                            nouns.append(noun_df2[noun_class])
                            noun_classes.append(noun_class)
                            pred_index.append(f_idx)

                            questions.append(qs_)
                            ys_true.append(noun_state.fillna(-1).values)
                            imgs.append(imgs_[i])
                            fut = pool.submit(get_detection_image, video_id, narration_id, f_idx, noun_class, h5_path)
                            futs.append(fut)

                    #     fut = pool.submit(get_detection_images, video_id, narration_id, f_idx, unique_noun_classes, h5_path)
                    #     futs.append(fut)
                    # dets = [img for f in futs for img in f.result()]
                    dets = [model.vis_processors["eval"](f.result()) for f in futs]

                    # predict frames
                    itm_pred = predict_itm(model, imgs, dets, questions)
                    # assert itm_pred.shape == (len(visible_index) * len(unique_noun_classes), questions)
                    # store predictions
                    for i in range(len(questions)):  # len(visible_index) * len(unique_noun_classes)
                        for j in range(len(questions[i])):  # questions
                            video_predictions.append({
                                'video_id': video_id,
                                'narration_id': narration_id,
                                'frame_index': pred_index[i],
                                'action_stage': action_stage,
                                'noun': nouns[i],
                                'noun_class': noun_classes[i],
                                'state': states[i][j],
                                'question': questions[i][j],
                                'text_match_true': ys_true[i][j],
                                'text_match_pred': itm_pred[i][j].tolist(),
                            })
                    #         print(questions[j], y_true[j], itm_pred[i, j].tolist())
                    acc = (np.concatenate(ys_true) == (np.concatenate(itm_pred) >= 0.5)).mean()
                    tqdm.tqdm.write(f'{narration_id} {",".join(sorted(set(nouns)))} {acc:.0%}')
                    
                    # # loop over nouns that we have detections for
                    # for noun_class in np.unique(class_ids):
                    #     if noun_class not in noun_df2:
                    #         # print('skip', noun_class)
                    #         continue
                    #     # sample N frames that we have detections for
                    #     visible_index = frame_index[current_frames & (class_ids == noun_class)]
                    #     if not len(visible_index):
                    #         continue
                    #     # noun = noun_df.key[noun_class]
                    #     noun = noun_df2[noun_class]

                    #     visible_index = np.random.choice(visible_index, min(n_frames, len(visible_index)))

                    #     # print()
                    #     # print(video_id, narration_id, noun, visible_index)

                    #     # load the assumed long-term state
                    #     try:
                    #         noun_state = cdf.loc[(action_stage, noun), predicate_cols].dropna()
                    #     except KeyError:
                    #         tqdm.tqdm.write(f"\n\n\n!!! skipping {(narration_id, action_stage, noun)} not in changes_df !!! \n\n\n")
                    #         continue
                    #         # raise
                    #         #noun_state = changes_df.loc[("pre", noun)].dropna()
                    #         #noun_state.values[:] = np.nan


                    #     # generate the model questions (and gt)
                    #     questions = [predicate_dict[c].format(noun, default='something') for c in noun_state.index]
                    #     questions = [f'is {q}' if not q.startswith('can ') else q for q in questions]
                    #     questions = [f'{noun} {q}' for q in questions]
                    #     state_gt = noun_state.index
                    #     y_true = noun_state.fillna(-1).values

                    #     # for i, ix in enumerate(visible_index):
                    #     #     meta['frame_index'].extend([ix]*len(questions))
                    #     #     meta['noun'].extend([noun]*len(questions))
                    #     #     meta['noun_class'].extend([noun_class]*len(questions))
                    #     #     meta['questions'].extend(questions)
                    #     #     meta['state'].extend(questions)
                    #     #     meta['text_match_true'].extend(y_true)

                    #     # print(questions)
                    #     # print(y_true)
                    #     # input()

                    #     # load frames
                    #     imgs = []
                    #     # dets = []
                    #     futs = []
                    #     for f_idx in visible_index:
                    #         img = Image.open(f'{DIR}/{video_id}/frame_{f_idx:010d}.jpg')
                    #         fut = pool.submit(get_detection_image, video_id, narration_id, f_idx, h5_path)
                    #         futs.append(fut)

                    #         # det = None
                    #         # for narr_id in [narration_id]:
                    #         #     ds = get_detections_h5(h5_file[narr_id], [frame_index], np.array(img))
                    #         #     if len(ds):
                    #         #         det = ds[0]
                    #         #         break
                    #         imgs.append(img)
                    #         # dets.append(det)
                    #     dets = [f.result() for f in futs]

                    #     # predict frames
                    #     itm_pred = predict_itm(model, imgs, dets, questions)
                    #     # print(itm_pred.shape)

                    #     # store predictions
                    #     for i in range(len(visible_index)):
                    #         for j in range(len(questions)):
                    #             video_predictions.append({
                    #                 'video_id': video_id,
                    #                 'frame_index': visible_index[i],
                    #                 'noun': noun,
                    #                 'noun_class': noun_class,
                    #                 'state': state_gt[j],
                    #                 'question': questions[j],
                    #                 'text_match_true': y_true[j],
                    #                 'text_match_pred': itm_pred[i, j].tolist(),
                    #             })
                    #     #         print(questions[j], y_true[j], itm_pred[i, j].tolist())
                    #     acc = (np.array(y_true)[None] == np.argmax(itm_pred, axis=-1)).mean()
                    #     tqdm.tqdm.write(f'{narration_id} {noun} {acc:.0%}')
                    #     # input()
            predictions.extend(video_predictions)
            video_predictions = pd.DataFrame(video_predictions)
            video_predictions.to_csv(vid_csv_path, index=False)
            # plot_state_predictions()
            # break
    # finally:
    #     if len(predictions) > 500:
    #         predictions = pd.DataFrame(predictions)
    #         predictions.to_csv(out_path, index=False)
    return predictions

    # predictions = []
    # predicates = list(set(predicate_dict) & set(changes_df.columns))
    # for video_id, vdf in tqdm.tqdm(changes_df.groupby('video_id')):
    #     for (narration_id, frame_index), fvdf in tqdm.tqdm(changes_df.groupby(['narration_id', 'frame_index'])):
    #         img = Image.open(f'{DIR}/{video_id}/frame_{frame_index:010d}.jpg')

    #         questions = []
    #         nouns_gt = []
    #         state_gt = []
    #         y_true = []
    #         for noun, r in fvdf.groupby('noun').last().iterrows():
    #             r = r[predicates].dropna()
    #             nouns_gt.extend([noun]*len(r))
    #             state_gt.extend(r.index)
    #             y_true.extend(r.values)
    #             for c, is_true in r.items():
    #                 predicate = predicate_dict[c]
    #                 questions.append(predicate.format(noun))
    #                 state_gt.append(c)
    #                 nouns_gt.append(noun)
    #                 y_true.append(is_true)

            
    #         det_img = img
    #         for narr_id in iidf.index.unique():
    #             ds = get_detections_h5(h5_file[narr_id], [frame_index], np.array(img))
    #             if len(ds):
    #                 det_img = draw_detections(img, ds[0], ds[0].data['labels'].tolist())
    #                 break

    #         Ximg = torch.stack([
    #             vis_processors["eval"](img), 
    #             vis_processors["eval"](det_img),
    #         ]).to(device)[None]
    #         Xtext = [text_processors["eval"](q) for q in question]

    #         itm_pred = model.itm({"image": Ximg, "text_match": [Xtext]})
    #         print(itm_pred.shape)
    #         itm_pred = itm_pred.cpu().numpy()[0, :, 1]
    #         print(itm_pred.shape)

    #         for i in range(len(questions)):
    #             predictions.append({
    #                 'video_id': video_id,
    #                 'frame_index': frame_index,
    #                 'question': questions[i],
    #                 'noun': nouns_gt[i],
    #                 'state': state_gt[i],
    #                 'text_match_true': y_true[i],
    #                 'text_match_pred': itm_pred[i],
    #             })

    # predictions = pd.DataFrame(predictions)
    # predictions.to_csv(out_path, index=False)

    # return predictions



def plot_state_predictions(
        y_true, y_pred, 
        labels=None, timesteps=None, 
        cmap='bone', 
        height=0.4,
        figsize=(15, 6),
        c_true='blue',
        c_pred='red',
        c_line='white',
    ):
    if y_pred is None:
        y_pred = y_true
    if y_pred.ndim == 3:
        y_pred = y_pred[:,:,1]
    T, M = y_pred.shape
    time_index = np.arange(T)

    # Plotting
    plt.figure(figsize=(min(max(15, T/50), 40), max(2, int(M*1/3))))
    cmap = plt.get_cmap(cmap)
    cmap.set_bad(color='#ffd65c')
    plt.imshow(y_pred.T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none')
    cbar = plt.colorbar()
    cbar.set_label('Prediction Probability (%)')
    cbar.set_ticks(np.linspace(0, 1, 11))
    cbar.set_ticklabels([f'{int(x*100)}%' for x in np.linspace(0, 1, 11)])

    # Ground truth
    lw = 0.4
    for m in range(M):
        ym_true = y_true[:, m]
        ym_pred = y_pred[:, m] > 0.5
        plt.fill_between(time_index - 0.5, m, m - ym_true * height/2, color=c_true, lw=lw, step='post', alpha=1, label='true' if m == 0 else "")
        plt.fill_between(time_index - 0.5, m, m + ym_pred * height/2, color=c_pred, lw=lw, step='post', alpha=1, label='pred' if m == 0 else "")
        plt.step(time_index - 0.5, m - ym_true * height/2, color=c_line, lw=lw, where='post')
        plt.step(time_index - 0.5, m + ym_pred * height/2, color=c_line, lw=lw, where='post')

    # Labels and title
    plt.xlabel('Time')

    # Example labels for Y-axis
    if labels is None:
        labels = np.arange(M)
    plt.yticks(range(M), labels)
    if timesteps is not None:
        t = np.linspace(0, len(timesteps)-1, 16).astype(int)
        plt.xticks(t, timesteps[t])
    
    # Add legend without blocking the imshow plot
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(0, 0))



# ---------------------------------------------------------------------------- #
#                                     Plot                                     #
# ---------------------------------------------------------------------------- #

import ipdb
@ipdb.iex
def main(annotation_dir='/scratch/bs3639/ego2023/epic-kitchens-100-annotations', h5_path='/scratch/bs3639/EKOS_val_3.h5', out_dir='plots'):
    # action_pddl, predicates = load_pddl_yaml(f'{annotation_dir}/EPIC_100_conditions.yaml')
    # ann_df = load_annotation_csv(f'{annotation_dir}/EPIC_100_validation.csv')
    # noun_df = load_nouns(annotation_dir)
    # h5_file = h5py.File(h5_path, 'r', libver='latest')
    change_df = expand_narrations_across_nouns()
    # change_df = ann_df.groupby('video_id').progress_apply(get_changes_for_video, actions=action_pddl).set_index('narration_id')
    # if input('>?'):from IPython import embed;embed()
    switch_df = calculate_windowed_switches(change_df)

    # model = load_model()
    # import pyinstrument
    # p = pyinstrument.Profiler()
    # try:
    # pred_df = predict_video(model, ann_df, change_df, predicates, h5_file, h5_path, noun_df)
        # video_ids = ann_df.video_id.unique()
        # mqdm.Bars.pool(
        #     predict_video,
        #     (
        #         mqdm.args(None, ann_df.loc[ann_df.video_id == i], change_df, predicates, None, h5_path, noun_df)
        #         for i in video_ids
        #     ),
        #     n_workers=5,
        # )
    # finally:
    #     # pd.concat([pd.read_csv(f) for f in glob.glob('output_video_predictions/P*.csv')]).to_csv('output_video_predictions/preds.csv', index=False)
    #     # p.print()
    #     pass
    if input('>?'):from IPython import embed;embed()
    return

    odf = df = get_stats()
    df = df.reset_index()
    pre_df = df.copy().assign(narration_id=df['narration_id'] + '_pre').drop(columns=[c for c in df.columns if 'post_' in c])
    post_df = df.copy().assign(narration_id=df['narration_id'] + '_post').drop(columns=[c for c in df.columns if 'pre_' in c])
    post_df = post_df.rename(columns={c: c.replace('post_', 'pre_') for c in post_df.columns})
    df = pd.concat([pre_df, post_df])
    df = df.rename(columns={c: c.replace('pre_', 'prepost_') for c in df.columns})
    df = df.set_index('narration_id')
    df['frame_gap_prior'] = df['frame_gap_prior'].clip(0, 1200) / 60
    df['frame_gap_after'] = df['frame_gap_after'].clip(0, 1200) / 60

    # result_path = 'lavis/output/results/ekos/ekos_ekos_pretrain_eval/20240516193/result/test_epic_kitchens_result_best.json'
    result_path = 'lavis/output/results/ekos/ekos_ekos_pretrain_eval/20240516194/result/test_epic_kitchens_result_best.json'
    rdf = pd.read_json(result_path)
    narration_id = rdf['narration_id']
    rdf = rdf.set_index(rdf['narration_id'] + '_' + rdf['context_mode'])
    itm_true = rdf['text_match_true'].values
    itm_pred = rdf['text_match_pred'].values

    rdf_data = json.load(open(result_path, 'r'))
    rdf_flat = []
    for row in rdf_data:
        for y_true, y_pred in zip(row['text_match_true'], row['text_match_pred']):
            d = dict(row)
            d['text_match_true'] = y_true
            d['text_match_pred'] = y_pred
            rdf_flat.append(d)
    rdf_flat = pd.DataFrame(rdf_flat)
    rdf_flat = rdf_flat.set_index(rdf_flat['narration_id'] + '_' + rdf_flat['context_mode'])
    df_flat = df.loc[rdf_flat.index]
    df_flat['text_match_true'] = rdf_flat['text_match_true']
    df_flat['text_match_pred'] = rdf_flat['text_match_pred']
    df_flat.to_csv('stats_preds.csv')

    df = df.loc[rdf.index]

    os.makedirs(out_dir, exist_ok=True)

    for c in df.columns:
        x = df[c]
        print(c, x.dtype)
        if x.dtype == object or '_post_' in c:
            continue
        plt.figure()
        plot_stuff(c, df[c].values, itm_true, itm_pred)
        plt.ylabel('F1')
        plt.savefig(f'{out_dir}/f1_{c}.png')

    plt.figure(figsize=(6, 2), dpi=300)
    ax=plt.subplot(121)
    # c='frame_gap_after'
    # plot_stuff(c, df[c].values, itm_true, itm_pred, 'Action Separation (sec)', p=1)
    plt.ylabel('F1 (object states)')
    # ax2=plt.subplot(132, sharey=ax)
    c='prepost_number_objects_bg'
    plot_stuff(c, df[c].values, itm_true, itm_pred, 'Background Object Count', p=1)
    # for tk in ax2.get_yticklabels(): tk.set_visible(False)
    ax2=plt.subplot(122, sharey=ax)
    c='overlapping_actions'
    plot_stuff(c, df[c].values, itm_true, itm_pred, 'Overlapping Actions', p=1)
    # plt.yticks([], [])
    for tk in ax2.get_yticklabels(): tk.set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/f1_density.png')

    # f1 = np.array([
    #     f1_score(np.array(yt), np.array(yp)[:, 1]>0.5) 
    #     for yt, yp in zip(itm_true, itm_pred)
    # ])

    plt.figure()
    df = pd.concat([df[[c for c in df.columns if c not in rdf.columns]], rdf], axis=1)
    scores = df.groupby('narration_id').apply(rate_example).sort_values('acc', ascending=False)
    scores.acc.plot.hist()
    plt.savefig(f'{out_dir}/acc.png')

    return 

    idf = pd.concat([
        odf.loc['P01_12_17':'P01_12_19'],
        odf.loc['P01_12_10':'P01_12_14'],
        odf.loc['P01_14_104':'P01_14_108'],
    ])
    captions = sorted(set(x for xs in [scores.loc[idf.index].pre, scores.loc[idf.index].post] for xx in xs for x in xx))
    print(idf)
    print(captions)
    # if input('>?'):from IPython import embed;embed()

    import torch
    from PIL import Image
    from lavis.models import load_model_and_preprocess
    from lavis.processors import load_processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.no_grad().__enter__()
    model, vis_processors, text_processors = load_model_and_preprocess("blip2_qformer2", "pretrain", device=device, is_eval=True)
    
    checkpoint = torch.load('lavis/output/results/epic_kitchens/epic_kitchens_pretrain/20240513160/checkpoint_best.pth')
    model.load_state_dict(checkpoint["model"], strict=False)

    DIR = '/rgb_frames'
    # if input('>?'):from IPython import embed;embed()

    h5_file = h5py.File('/scratch/bs3639/EKOS_val.h5', 'r', libver='latest')
    idf = pd.concat([
        # odf.loc['P01_12_17':'P01_12_19'],
        # odf.loc['P01_12_10':'P01_12_14'],
        odf.loc['P01_14_105':'P01_14_108'],
        # odf.loc['P01_14_104':'P01_14_108'],
        # odf.loc['P08_14_11':'P08_14_19'],
        # odf.loc['P01_12_45':'P01_12_48'],
        
    ])
    CAPTIONS = {
        # 'P01_12': [
        #     'drawer is open', 'drawer is not open',
        #     'cupboard is open', 'cupboard is not open',
        #     'cupboard is not closed', 'cupboard is closed',
        #     'cutting board is in cupboard', 'cutting board is not in cupboard',
        # ],
        # 'P01_14': [
        #     # 'tap is on', 'tap is off', 
        #     # 'tap is touching hand', 'tap is not touching hand',
        #     # 'kettle is touching hand', 'kettle is not touching hand',
        #     # 'kettle is on surface', 'kettle is not on surface',
        #     'tap is turned on', 'tap is turned off',
        #     'kettle is being held', 'kettle is not being held',
        #     'kettle is on a surface', 'kettle is not on a surface',
        #     'kettle is turned on', 'kettle is turned off',
        # ],
        'P01_14': [
            'kettle is open',
            'kettle is turned on',
            'kettle is touching hand',
            'cupboard is open',
            
        ],
    }
    for video_id, iidf in idf.groupby('video_id'):
        print(video_id)
        iidf = iidf.loc[iidf.index.intersection(scores.index)]
        s = scores.loc[iidf.index]
        captions = CAPTIONS.get(video_id) or sorted(set(x for xs in [s.pre, s.post] for xx in xs for x in xx if not any(s in x for s in ['holdable'])))
        frame_idx = []
        preds = []
        txt = []
        for caption in captions:
            txt.append(text_processors["eval"](caption))
        f_min = iidf.start_frame.min() - 60
        f_max = iidf.stop_frame.max() + 60

        frame_ids = np.linspace(f_min, f_max, 120).astype(int)
        for i in tqdm.tqdm(frame_ids, desc=video_id):
            img = Image.open(f'{DIR}/{video_id}/frame_{i:010d}.jpg')
            for nid in iidf.index.unique():
                ds = get_detections_h5(h5_file[nid], [i], np.array(img))
                if len(ds):
                    break
            det_img = draw_detections(img, ds[0], ds[0].data['labels'].tolist())
            img = torch.stack([
                vis_processors["eval"](img), 
                vis_processors["eval"](det_img),
            ])
            itm_pred = model.itm({"image": img.to(device)[None], "text_match": [txt]})
            frame_idx.append(i)
            preds.append(itm_pred.cpu().numpy()[0, :, 1])
        frame_idx = np.array(frame_idx)
        preds = np.array(preds)

        plt.figure(figsize=(15, len(captions)*0.3 + 2))
        ix = np.argsort(preds.mean(0))
        plt.imshow(preds[:, ix].T, aspect='auto', extent=[frame_idx.min(), frame_idx.max(), -0.5, len(captions)-0.5], cmap='cubehelix')
        for i, (_, r) in enumerate(iidf.iterrows()):
            plt.axvline(r.start_frame, c='white')
            plt.axvline(r.stop_frame, c='red', linestyle='dotted')
            plt.text(r.start_frame, len(captions) + (i%3)*1.2, r.narration)
        plt.yticks(range(len(captions)), np.array(captions)[ix])
        plt.tight_layout()
        plt.savefig(f'{out_dir}/video_{iidf.index[0]}.png')

    # plt.set_cmap('Pastel1')
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set1.colors)
    plt.figure(figsize=(8, 2), dpi=300)

    for i in range(preds.shape[1]):
        p = preds[:, i]
        p = gaussian_filter1d(p, 1)
        plt.plot(frame_idx, p, label=captions[i].replace('is ', '').replace('touching ', 'in ').replace('turned ', ''), linewidth=2)

    # for i, (_, r) in enumerate(iidf.iterrows()):
    #     plt.axvline(r.start_frame, c='black')
    #     plt.axvline(r.stop_frame, c='red', linestyle='dotted')
    #     plt.text(r.start_frame, 1 + (i%2)*0.05, r.narration)
    plt.ylim([0, 1])
    plt.tick_params(
        axis='x',         
        which='both',     
        bottom=False,     
        top=False,        
        labelbottom=False)
    # plt.tick_params(
    #     axis='y',         
    #     which='both',     
    #     right=False,     
    #     left=False,        
    #     labelbottom=False)
    plt.yticks([0, 0.5, 1], [0, 0.5, 1])
    import matplotlib.ticker as mtick
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

    plt.legend(bbox_to_anchor=(0.44,0.35))
    plt.tight_layout()
    plt.savefig(f'{out_dir}/video_line_{iidf.index[0]}.png')

    from IPython import embed;embed()

def PIL_load(f, shape=None): 
    im = Image.open(f) 
    if shape is not None: 
        im.draft('RGB',tuple(shape)) #(1008,756)
    return im

def rate_example(df):
    pre = df[df.context_mode == 'pre'].iloc[0]
    post = df[df.context_mode == 'post'].iloc[0]
    yt = np.array(pre.text_match_true + post.text_match_true)
    yp = np.array(pre.text_match_pred + post.text_match_pred)[:, 1]>0.5
    acc = (yt == yp).mean()

    return pd.Series({
        'acc': acc,
        'verb': pre.verb,
        'noun': pre.noun,
        'narration': pre.narration,
        'pre': pre.text_match,
        'post': post.text_match,
    })


def plot_stuff(c, x, itm_true, itm_pred, *a, **kw):
    # drop nans
    mask = ~np.isnan(x)
    itm_true = itm_true[mask]
    itm_pred = itm_pred[mask]
    x = x[mask]
    print(c, (x == 0).mean())


    if not len(x):
        print("No data for ", c)
        return 
    g=plt.GridSpec(4, 1)
    ax = plt.subplot(g[0])
    plt.axvline(x.min(), color='black', label=f'{x.min():.2g}')
    plt.axvline(x.max(), color='black', label=f'{x.max():.2g}')
    plt.hist(x, color='red', bins=100)
    plt.yscale('log')
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.legend()

    plt.subplot(g[1:], sharex=ax)
    # if (np.mod(x) == 0).all():
    # else:
    # , logx=any(k in c for k in [
    #     'displacement',
    #     'area'
    # ])
    plot_discrete(c, x, itm_true, itm_pred, *a, **kw)
    # if any(k in c for k in [
    #     'displacement',
    #     'area'
    # ]):
    #     plt.xscale('log')
        # plt.ylim([max(-1000, plt.gca().get_ylim()[0])])
    # plot_cont(c, x, itm_true, itm_pred, *a, **kw)

def plot_cont(c, x, itm_true, itm_pred, title=None, p=1, bin_size=1):
    # flatten arrays
    x = np.array([xi for i, xi in enumerate(x) for _ in range(len(itm_true[i]))])
    itm_true = np.array([xi for xs in itm_true for xi in xs])
    itm_pred = np.array([xi for xs in itm_pred for xi in xs])
    assert itm_true.ndim == 1
    assert itm_true.shape + (2,) == itm_pred.shape

    y = itm_pred[np.arange(len(itm_true)), itm_true]
    plt.scatter(x, y, c='skyblue', alpha=0.05)
    # plt.hist2d(x, y, bins=100, norm=colors.LogNorm(), cmap="bone_r")

    xa = np.linspace(x.min(), x.max(), 100)
    reg = Polynomial.fit(x, y, min(p, len(x)-1))#
    ya = reg(xa)
    ya[ya < np.nanmin(y) - 0.05] = np.nan
    ya[ya > np.nanmax(y) + 0.05] = np.nan
    plt.plot(xa, ya, color='k', linestyle='dashed')

    plt.xlabel(title or c.replace('_', ' ').replace(' pre ', ' ').replace('num', 'number of').title())


def plot_discrete(c, x, itm_true, itm_pred, title=None, p=1, logx=False, nbins=32):
    xmin = x.min()
    xmax = x.max()
    if not logx and (np.mod(x, 1) == 0).all() and xmin >= 0 and xmax < nbins * 2: # check for integers
        bin_size = 1
    else:
        bin_size = (xmax-xmin)/nbins
    # if logx:
    #     logxmin = np.log(1e-3)
    #     logxmax = np.log(xmax - xmin + 1e-3)
    #     bin_size = (logxmax - logxmin)/nbins
    #     bins = np.exp(np.arange(logxmin, logxmax, bin_size)) + xmin
    # else:
    bins = np.arange(xmin, xmax + bin_size, bin_size)
    xd = np.digitize(x, bins) - 1
    y = []
    ns = []
    for i in range(len(bins)):
        yt = np.array([y for ys in itm_true[xd == i] for y in ys])
        yp = np.array([y for ys in itm_pred[xd == i] for y in np.array(ys)[:, 1]>0.5])
        ns.append(len(yt))
        y.append(f1_score(yt, yp) if len(yt) else np.nan)
    y = np.array(y)
    ns = np.array(ns)
    mask = ~np.isnan(y) & (ns > 50)
    bins = bins[mask]
    ns = ns[mask]
    y = y[mask]
    w = ns / np.sum(ns)
    print(w)

    plt.scatter(bins, y, c='red', s=w * 100 + 3)
    x_bins = bins
    # if logx:
    #     x_bins = np.log(bins)
    #     plt.xscale('symlog')
        
    xa = np.linspace(x_bins.min(), x_bins.max(), 100)
    if len(y) > 1:
        # w = np.exp(ns - np.max(ns)) / np.maximum(np.sum(np.exp(ns - np.max(ns))), 1e-7)
        
        cs = ['k','b','g']
        for pi in range(min(2, len(x_bins)-1)):
            reg = Polynomial.fit(x_bins, y, pi+1, w=w)#
            ya = reg(xa)
            ya[ya < np.nanmin(y) - 0.05] = np.nan
            ya[ya > np.nanmax(y) + 0.05] = np.nan
            plt.plot(xa, ya, color=cs[pi], linestyle='dashed')
    # plt.ylim(np.nanmin(y), np.nanmax(y))
    plt.xlabel(title or c.replace('_', ' ').replace(' pre ', ' ').replace(' prepost ', ' ').replace('num_', 'number of ').title())
    

def plot_videos(df):
    pass



if __name__ == '__main__':
    import fire
    fire.Fire(main)