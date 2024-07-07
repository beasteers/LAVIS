import os
import random
import tqdm
tqdm.tqdm.pandas()
import h5py
import pandas as pd


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
    df['noun'] = df['noun_norm']
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


def get_event_stats(row, h5, buffer=10):
    g = h5[row.narration_id]
    frame_ids = g['frame_index'][()]
    class_ids = g['class_ids'][()]
    class_id = row.noun_class

    # time boundaries
    pre_0 = row.start_frame - buffer
    pre_1 = row.start_frame
    mid_0 = row.start_frame
    mid_1 = row.stop_frame
    post_0 = row.stop_frame
    post_1 = row.stop_frame + buffer

    # filter frames by time boundaries
    mn_frames = frame_ids[class_ids == class_id]
    pre_mn_frames = frame_ids[(class_ids == class_id) & (frame_ids >= pre_0) & (frame_ids < pre_1)]
    mid_mn_frames = frame_ids[(class_ids == class_id) & (frame_ids >= mid_0) & (frame_ids < mid_1)]
    post_mn_frames = frame_ids[(class_ids == class_id) & (frame_ids >= post_0) & (frame_ids < post_1)]
    
    # store frames
    row['main_noun_frames'] = list(mn_frames)
    row['n_main_noun_frames'] = len(mn_frames)
    row['main_noun_pre_frames'] = list(pre_mn_frames)
    row['n_main_noun_pre_frames'] = len(pre_mn_frames)
    row['main_noun_mid_frames'] = list(mid_mn_frames)
    row['n_main_noun_mid_frames'] = len(mid_mn_frames)
    row['main_noun_post_frames'] = list(post_mn_frames)
    row['n_main_noun_post_frames'] = len(post_mn_frames)
    return row


def sample_actions(df, noun_ratio, n):
    df = df.sample(n=min(int(n*noun_ratio[df.name]), len(df)))
    return df

def sample_frame(row, h5, key):
    g = h5[row.narration_id]
    frame_id = random.choice(row[key])
    mask = (g['frame_index'][()] == frame_id) & (g['class_ids'][()] == row.noun_class)
    
    row['frame_id'] = frame_id
    row['polygons'] = g['segments'][mask]
    return row


def extract(annotation_path, h5_path, p_actions=0.1):
    df = load_annotation_csv(annotation_path)
    total_count = len(df)
    noun_count = df.noun.value_counts()
    noun_ratio = noun_count / noun_count.sum()
    h5 = h5py.File(h5_path, 'r', libver='latest')

    # compute per-event stats
    df = df.progress_apply(get_event_stats, h5=h5, axis=1)
    df.to_csv('mturk_stat.csv')
    # df = pd.read_csv('mturk_stat.csv')
    # df['main_noun_pre_frames'] = df.main_noun_pre_frames.apply(eval)
    # df['main_noun_mid_frames'] = df.main_noun_mid_frames.apply(eval)
    # df['main_noun_post_frames'] = df.main_noun_post_frames.apply(eval)

    # filter out actions with no noun seg
    df = df[df.n_main_noun_frames > 0]

    # filter out actions with no noun seg in each portion
    pre_df = df[df.n_main_noun_pre_frames > 0].assign(temporal_loc='pre')
    mid_df = df[df.n_main_noun_mid_frames > 0].assign(temporal_loc='mid')
    post_df = df[df.n_main_noun_post_frames > 0].assign(temporal_loc='post')

    # sample 10% of actions proportionally across nouns and pre/mid/post
    # then sample a frame from the available frames
    n_actions = p_actions * 5 * total_count
    df = pd.concat([
        pre_df.groupby('noun').apply(sample_actions, noun_ratio=noun_ratio, n=n_actions * 0.4).progress_apply(sample_frame, h5=h5, key='main_noun_pre_frames', axis=1),
        mid_df.groupby('noun').apply(sample_actions, noun_ratio=noun_ratio, n=n_actions * 0.2).progress_apply(sample_frame, h5=h5, key='main_noun_mid_frames', axis=1),
        post_df.groupby('noun').apply(sample_actions, noun_ratio=noun_ratio, n=n_actions * 0.4).progress_apply(sample_frame, h5=h5, key='main_noun_post_frames', axis=1),
    ])

    # write out result
    df = df[['narration_id', 'temporal_loc', 'noun', 'frame_id', 'polygons']]
    return df

import ipdb
@ipdb.iex
def main(annotation_dir='/scratch/bs3639/ego2023/epic-kitchens-100-annotations', h5_path='/scratch/bs3639/EKOS_val.h5', p_actions=0.1):
    train_df = extract(f'{annotation_dir}/EPIC_100_train.csv', '/scratch/bs3639/EKOS_train.h5').assign(split='train')
    val_df = extract(f'{annotation_dir}/EPIC_100_validation.csv', '/scratch/bs3639/EKOS_val.h5').assign(split='val')
    df = pd.concat([train_df, val_df])

    # write out result
    df.to_csv('mturk_results.csv')
    print(df.groupby('narration_id').temporal_loc.unique().apply(lambda x: tuple(sorted(x))).value_counts())
    if input('>?'):from IPython import embed;embed()

if __name__ == '__main__':
    import fire
    fire.Fire(main)