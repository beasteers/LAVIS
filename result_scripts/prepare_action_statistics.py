import os
import sys
sys.path.append('.')
import tqdm
tqdm.tqdm.pandas()
import orjson
import h5py
from PIL import Image
import numpy as np
from numpy.polynomial import Polynomial
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix, precision_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from lavis.common.predicate_utils.masks import get_detections, draw_detections, get_detections_h5
# from lavis.datasets.ekos_datasets import load_annotation_csv, load_nouns, load_verbs

HAND_IDS = [11,300,301,303,304]

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
    df['frame_gap_prior'] = st - nearest_before
    df['frame_gap_after'] = nearest_after - et
    df['overlapping_actions'] = overlap
    return df

def get_narration_objects(narration_id, noun_class, all_noun_classes, start_frame, stop_frame, h5f):
    g = h5f[narration_id]
    # segments = g['segments']
    # names = g['names']
    class_ids = g['class_ids'][()]
    frame_index = g['frame_index'][()]
    pre_idx = frame_index[(frame_index > start_frame - 30) & (frame_index < start_frame + 30)]
    post_idx = frame_index[(frame_index > stop_frame - 30) & (frame_index < stop_frame + 30)]

    return {
        'narration_id': narration_id,
        'num_objects': len(np.unique(class_ids)),
        'num_pre_objects': len(np.unique(class_ids[np.isin(frame_index, pre_idx)])),
        'num_pre_active_objects': len(np.unique(class_ids[
            np.isin(frame_index, pre_idx) & 
            np.isin(class_ids, [noun_class])
        ])),
        'num_pre_bg_objects': len(np.unique(class_ids[
            np.isin(frame_index, pre_idx) & 
            ~np.isin(class_ids, [noun_class] + HAND_IDS)
        ])),
        'num_pre_hands': len(np.unique(class_ids[
            np.isin(frame_index, pre_idx) & 
            np.isin(class_ids, HAND_IDS)
        ])),
        'num_post_objects': len(np.unique(class_ids[np.isin(frame_index, post_idx)])),
        'num_post_active_objects': len(np.unique(class_ids[
            np.isin(frame_index, post_idx) & 
            np.isin(class_ids, [noun_class])
        ])),
        'num_post_bg_objects': len(np.unique(class_ids[
            np.isin(frame_index, post_idx) & 
            ~np.isin(class_ids, [noun_class] + HAND_IDS)
        ])),
        'num_post_hands': len(np.unique(class_ids[
            np.isin(frame_index, post_idx) & 
            np.isin(class_ids, HAND_IDS)
        ])),
    }


def get_stats(annotation_dir='/scratch/bs3639/ego2023/epic-kitchens-100-annotations', h5_path='/scratch/bs3639/EKOS_val.h5'):
    df = load_annotation_csv(f'{annotation_dir}/EPIC_100_validation.csv')
    act_df = df.groupby('video_id').apply(process_video_actions).set_index('narration_id')
    # act_df.to_csv("action_stats.csv")
    # for c in act_df.columns:
    #     print(c)
    #     plt.figure()
    #     act_df[c].plot.hist(bins=60)
    #     plt.title(c)
    #     plt.savefig(f'hist_{c}.png')

    h5_file = h5py.File(h5_path, 'r', libver='latest')
    ann_df = pd.DataFrame([
        get_narration_objects(r.narration_id, r.noun_class, r.all_noun_classes, r.start_frame, r.stop_frame, h5_file)
        for i, r in tqdm.tqdm(df.iterrows(), total=len(df))
    ]).set_index('narration_id')
    # ann_df.to_csv("ann_stats.csv")
    # for c in ann_df.columns:
    #     print(c)
    #     plt.figure()
    #     ann_df[c].plot.hist()
    #     plt.title(c)
    #     plt.savefig(f'hist_{c}.png')
    # if input('>?'):from IPython import embed;embed()
    # [['narration_id', 'narration', 'verb', 'noun']]
    ddf = df.set_index('narration_id')
    ddf = pd.concat([ddf, act_df, ann_df], axis=1)
    ddf.to_csv('stats.csv')
    return ddf



# ---------------------------------------------------------------------------- #
#                                     Plot                                     #
# ---------------------------------------------------------------------------- #

import ipdb
@ipdb.iex
def main():
    # df = get_stats()
    df = pd.read_csv('stats.csv')
    odf = df.set_index('narration_id').sort_index()
    pre_df = df.copy().assign(narration_id=df['narration_id'] + '_pre').drop(columns=[c for c in df.columns if 'post_' in c])
    post_df = df.copy().assign(narration_id=df['narration_id'] + '_post').drop(columns=[c for c in df.columns if 'pre_' in c])
    post_df = post_df.rename(columns={c: c.replace('post_', 'pre_') for c in post_df.columns})
    df = pd.concat([pre_df, post_df])
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

    df = df.loc[rdf.index]

    for c in df.columns:
        x = df[c]
        print(c, x.dtype)
        if x.dtype == object or '_post_' in c:
            continue
        plt.figure()
        plot_stuff(c, df[c].values, itm_true, itm_pred)
        plt.ylabel('F1')
        plt.savefig(f'f1_{c}.png')

    plt.figure(figsize=(6, 2), dpi=300)
    ax=plt.subplot(121)
    # c='frame_gap_after'
    # plot_stuff(c, df[c].values, itm_true, itm_pred, 'Action Separation (sec)', p=1)
    plt.ylabel('F1 (object states)')
    # ax2=plt.subplot(132, sharey=ax)
    c='num_pre_bg_objects'
    plot_stuff(c, df[c].values, itm_true, itm_pred, 'Background Object Count', p=1)
    # for tk in ax2.get_yticklabels(): tk.set_visible(False)
    ax2=plt.subplot(122, sharey=ax)
    c='overlapping_actions'
    plot_stuff(c, df[c].values, itm_true, itm_pred, 'Overlapping Actions', p=1)
    # plt.yticks([], [])
    for tk in ax2.get_yticklabels(): tk.set_visible(False)
    plt.tight_layout()
    plt.savefig(f'f1_density.png')

    # f1 = np.array([
    #     f1_score(np.array(yt), np.array(yp)[:, 1]>0.5) 
    #     for yt, yp in zip(itm_true, itm_pred)
    # ])

    plt.figure()
    df = pd.concat([df[[c for c in df.columns if c not in rdf.columns]], rdf], axis=1)
    scores = df.groupby('narration_id').apply(rate_example).sort_values('acc', ascending=False)
    scores.acc.plot.hist()
    plt.savefig('acc.png')

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
    plt.savefig(f'video_{iidf.index[0]}.png')

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
plt.savefig(f'video_line_{iidf.index[0]}.png')

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


def plot_stuff(c, x, itm_true, itm_pred, title=None, p=1, bin_size=1):
    xmin = x.min()
    xmax = x.max()
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
    print(ns)

    plt.scatter(bins, y, c='skyblue')
    xa = np.linspace(bins.min(), bins.max(), 100)
    if len(y) > 1:
        reg = Polynomial.fit(bins, y, min(p, len(bins)-1), w=np.log(ns))#
        ya = reg(xa)
        ya[ya < np.nanmin(y) - 0.05] = np.nan
        ya[ya > np.nanmax(y) + 0.05] = np.nan
        plt.plot(xa, ya, color='k', linestyle='dashed')
    # plt.ylim(np.nanmin(y), np.nanmax(y))
    plt.xlabel(title or c.replace('_', ' ').replace(' pre ', ' ').replace('num', 'number of').title())
    


def plot_videos(df):
    pass



if __name__ == '__main__':
    import fire
    fire.Fire(main)