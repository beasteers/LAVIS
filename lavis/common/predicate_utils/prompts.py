import os
import json
import tqdm
import random
import numpy as np

'''

- B: QA_describe_predicates: Correct Predicates: Which of the following predicates apply to the "fridge"? not open. not closed. is-openable.
    - S: Negative attributes: What of the following predicates do not apply to the fridge?     # no seg
- B: QA_action_before_after
    - Implicit Pre-conditions: Based on this image, is it possible to execute the action 'opening the fridge' at this moment?
    - Implicit Post-conditions: Based on this image, has "opening the fridge" been completed?
- B: QA_action_complete: Action Completion: Based on this image, has "opening the jar" been COMPLETED?
- S: QA_object_id: Object ID: what is object number 3?                                                   # segm (no-seg)
- S: QA_main_object_id: Main Object ID: what is object number 3? (always main noun)                                 # segm (no-seg)
- B*: QA_yes_no_predicate: Direct yes/no: is object 4 on a surface?                                              # seg
- S*: QA_object_from_predicate: Affordances: What numbered object is openable?                                        # seg 
    - S: Contrastive: What numbered objects are on a surface but not active?                   # seg
    - S: Composite: What numbered objects are on a surface and active?                         # seg


object_id
main_object_id
object_from_predicate
'''


def PRE_true_false_statements(ann, detection_labels=None, predicate_freq=None, n_pos=3, n_neg=6, random_prompt_mix=True):
    noun = ann['noun']
    all_nouns = ann['all_nouns']
    # state = ann['state']
    action = ann['action']
    noun_dict = action.var_dict(*all_nouns)
    candidate_state = action.get_state(action.vars[0], ann['pre_post'])
    noun_dict = action.var_dict(*all_nouns)
    pos_states = sample_states(candidate_state, predicate_freq, n_pos)
    neg_states = [s.inv() for s in sample_states(candidate_state, predicate_freq, n_neg)]
    pos_text = [f'{noun} is {s.format(**noun_dict)}' for s in pos_states]
    neg_text = [f'{noun} is {s.format(**noun_dict)}' for s in neg_states]
    # print(action, noun, all_nouns)
    # print(pos_text)
    # print(neg_text)
    # input()
    return pos_text, neg_text


def QA_describe_predicates(ann, detection_labels=None, predicate_freq=None, random_prompt_mix=True):
    # np.random.seed(12345)
    noun = ann['noun']
    all_nouns = ann['all_nouns']
    # state = ann['state']
    action = ann['action']
    candidate_state = action.get_state(action.vars[0], ann['pre_post'])
    noun_dict = action.var_dict(*all_nouns)

    # use at most 5 states
    # random.shuffle(candidate_state)
    # candidate_state = candidate_state[:8]
    candidate_state = sample_states(candidate_state, predicate_freq, 5)

    state_input = []
    state_output = []
    for s in candidate_state:
        # if len(s.vars) > len(noun_dict):
        #     continue
        if '?' in s.format(**noun_dict):
            continue
        state_input.append(s.flip(np.random.rand() < 0.3).format(**noun_dict))
        state_output.append(s.format(**noun_dict))

    state_list = " ".join(set(f'{x}.' for x in state_input))
    text_output = " ".join(set(f'{x}.' for x in state_output))

    objs = ''
    if detection_labels is not None:
        objs = f"Objects in scene: {', '.join(f'{i}: {o}' for i, o in enumerate(detection_labels))}.  "

    text_input = [
        f'Based on the image, which of the following predicates apply to the "{noun}"? {state_list}',
        f'Which of the following predicates apply to the "{noun}"? {state_list}',
        f'Which of the following predicates describe the "{noun}"? {state_list}',
        f'Which of the following apply to the "{noun}"? {state_list}',
        f'Which of the following describe the "{noun}"? {state_list}',
    ]
    text_input = random.choice(text_input) if random_prompt_mix else text_input[0]
    text_input = f'{objs}{text_input} Answer: '
    
    return text_input.strip(), text_output.strip()



def QA_object_id(ann, detection_labels=None, predicate_freq=None, random_prompt_mix=True):
    text_output = 'none'
    object_index = '0'
    if detection_labels:
        object_index = np.random.randint(len(detection_labels))
        text_output = detection_labels[object_index]

    text_input = [
        f'What is object number {object_index}?',
        f'What object is labeled as {object_index}?',
        f'What is object #{object_index}?',
    ]
    text_input = random.choice(text_input) if random_prompt_mix else text_input[0]
    text_input = f'Question: {text_input} Answer: '

    return text_input.strip(), text_output.strip()


def QA_main_object_id(ann, detection_labels=None, predicate_freq=None, random_prompt_mix=True):
    noun = ann['noun']
    text_output = 'none'
    object_index = '0'
    if detection_labels and noun in detection_labels:
        object_index = detection_labels.index(noun)
        text_output = noun
    
    text_input = [
        f'What is object number {object_index}?',
        f'What object is labeled as {object_index}?',
        f'What is object #{object_index}?',
    ]
    text_input = random.choice(text_input) if random_prompt_mix else text_input[0]
    text_input = f'Question: {text_input} Answer: '

    return text_input.strip(), text_output.strip()



def QA_action_before_after(ann, detection_labels=None, predicate_freq=None, random_prompt_mix=True):
    is_pre = ann['pre_post'] == 'pre'
    flipped = np.random.rand() < 0.5

    narration = ann['narration']

    if is_pre != flipped:  # pre and not flipped (or post and flipped)
        text_input = [
            f"Based on this image, is it possible to execute the action '{narration}' at this moment?",
            f'Can the action "{narration}" be performed now?',
            f'Can I perform the action "{narration}"?',
            f'Based on the current state of the objects in the scene, could the action "{narration}" be performed?',
            f'Is it likely that the action "{narration}" will be performed now?',
            f'Are the preconditions of the action "{narration}" satisfied?',
            f'Is it likely that the next action will be "{narration}"?',
            f"Considering the current scenario depicted, is it feasible to carry out the action '{narration}'?",
            f"Is the execution of the action '{narration}' possible at this moment?",
            f"Given the present conditions, am I capable of performing the action '{narration}'?",
            f"Observing the current arrangement of objects, is the action '{narration}' practicable?",
            f"Does the immediate context suggest that the action '{narration}' is about to occur?",
            f"Are all necessary conditions met for initiating the action '{narration}'?",
            f"Considering the setup, is it probable that '{narration}' is the next action?",
        ]
    else:
        text_input = [
            f'Based on this image, has "{narration}" been completed?',
            f'Based on this image, has the action "{narration}" been completed?',
            f'Has "{narration}" been completed?',
            f'Have I completed the action "{narration}"?',
            f'Are the postconditions of the action "{narration}" satisfied?',
            f'Based on this image, was the last action performed "{narration}"?',
            f'Based on this image, has the action "{narration}" been performed?',
            f"Does this image indicate that '{narration}' has already been carried out?",
            f"In light of this image, can we deduce that the action '{narration}' has concluded?",
            f"Has the action '{narration}' reached completion?",
            f"Reflecting on my actions, have I successfully executed '{narration}'?",
            f"Do the outcomes evident in this image fulfill the action '{narration}'s postconditions?",
            f"From the evidence presented, was '{narration}' the preceding action?",
            f"Does this visual suggest that the action '{narration}' has been accomplished?",
        ]
    text_input = random.choice(text_input) if random_prompt_mix else text_input[0]
    text_input = f'{text_input} Answer: '

    if detection_labels is not None and np.random.rand() < 0.2:
        objs = f"Objects in scene: {', '.join(f'{i}: {o}' for i, o in enumerate(detection_labels))}.  "
        text_input = f'{objs}{text_input}'

    text_output = 'no' if flipped else 'yes'

    return text_input.strip(), text_output.strip()



def QA_action_complete(ann, detection_labels=None, predicate_freq=None, random_prompt_mix=True):
    complete = ann['pre_post'] == 'post'
    narration = ann['narration']
    text_input = f'Based on this image, has "{narration}" been COMPLETED?'
    text_input = f'{text_input} Answer: '

    if detection_labels is not None and np.random.rand() < 0.2:
        objs = f"Objects in scene: {', '.join(f'{i}: {o}' for i, o in enumerate(detection_labels))}.  "
        text_input = f'{objs}{text_input}'

    text_output = 'yes' if complete else 'no'

    return text_input.strip(), text_output.strip()



def QA_yes_no_predicate(ann, detection_labels=None, predicate_freq=None, random_prompt_mix=True):
    noun = ann['noun']
    all_nouns = ann['all_nouns']
    action = ann['action']
    states = action.get_state(action.vars[0], ann['pre_post'])
    noun_dict = action.var_dict(*all_nouns)
    
    picked_state = sample_states(states, predicate_freq)

    # flip positive to negative
    flipped = np.random.rand() < 0.5
    if flipped:
        picked_state = picked_state.flip(not picked_state)

    # answer
    text_output = 'yes' if bool(picked_state) != flipped else 'no'

    # Get object description list
    objs = ''
    noun_index = -1
    if detection_labels is not None:
        objs = f"Objects in scene: {', '.join(f'{i}: {o}' for i, o in enumerate(detection_labels))}.  "
        if noun in detection_labels:
            noun_index = detection_labels.index(noun)
    
    state_str = picked_state.format(**noun_dict)
    text_input = [
        f'is {noun} {state_str}?',
        f'is {noun} {state_str}? (yes/no)',
        f'Does "{state_str}" describe the {noun}?',
        *([
            f'is #{noun_index} {state_str}?',
            f'is object #{noun_index} {state_str}?',
            f'Does "{state_str}" apply to object #{noun_index}?',
        ] if noun_index >= 0 else []),
    ]
    text_input = random.choice(text_input) if random_prompt_mix else text_input[0]
    text_input = f'{objs}Question: {text_input} Answer: '
    return text_input.strip(), text_output.strip()


def QA_object_from_predicate(ann, detection_labels=None, predicate_freq=None, random_prompt_mix=True):
    noun = ann['noun']
    all_nouns = ann['all_nouns']
    action = ann['action']
    states = action.get_state(action.vars[0], ann['pre_post'])
    states = sample_states(states, predicate_freq, np.random.choice([2, 3], p=[0.75, 0.25]))
    noun_dict = action.var_dict(*all_nouns)

    # Get object description list
    objs = ''
    if detection_labels is not None and np.random.rand() < 0.1:
        objs = f"Objects in scene: {', '.join(f'{i}: {o}' for i, o in enumerate(detection_labels))}.  "
    
    text_output = 'none'
    if noun in detection_labels:
        text_output = str(detection_labels.index(noun))

    states_str = join_and([s.format(**noun_dict) for s in states])
    text_input = random.choice([
        f"What numbered objects are {states_str}?",
        f"Which detected objects are {states_str}?",
        f"Which listed objects are {states_str}?",
        f"Which shown objects are {states_str}?",
        f"Which detected objects are {states_str}?",
    ])
    text_input = random.choice(text_input) if random_prompt_mix else text_input[0]
    text_input = f'{objs}Question: {text_input} Answer: '
    return text_input.strip(), text_output.strip()






def sample_states(states, freq, n=None):
    weights = None
    if freq is not None:
        weights = np.array([freq.get(k, 1) for k in states])
        weights = weights / weights.sum()
    # n = min(n, len(states)) if n else None
    return np.random.choice(list(states), n, replace=n is not None and n > len(states), p=weights)

def join_and(states):
    if len(states) == 2:
        return ' and '.join(states)
    if len(states) > 2:
        states = states[:-1] + [f'and {states[-1]}']
    return ', '.join(map(str, states))



def QA_mix(variants=None):
    if variants:
        unknown = set(variants) - set(PROMPTS)
        assert not unknown, f"Unknown variants: {unknown}"
    def QA_mix(ann, detection_labels=None, **kw):
        qas = [
            'describe_predicates', 
            'yes_no_predicate',
            'action_before_after',
            'action_complete',
        ]

        # required numbered main object
        if detection_labels:
            qas.extend([
                'object_id',
            ])
            if ann['noun'] in detection_labels:
                qas.extend([
                    'main_object_id', 
                    'object_from_predicate',
                ])

        if variants:
            qas = [q for q in qas if q in variants]
        # TODO: handle case where qas is empty - go to next sample
        kw.setdefault('random_prompt_mix', False)
        return PROMPTS[random.choice(qas)](ann, detection_labels, **kw)
    return QA_mix


PROMPTS = {
    'describe_predicates': QA_describe_predicates, 
    'yes_no_predicate': QA_yes_no_predicate,
    'action_before_after': QA_action_before_after,
    'action_complete': QA_action_complete,
    'object_id': QA_object_id,
    'main_object_id': QA_main_object_id, 
    'object_from_predicate': QA_object_from_predicate,
}


def get_prompt_function(prompt):
    if callable(prompt):
        return prompt
    if isinstance(prompt, str):
        print("QA Task:", prompt)
        return PROMPTS[prompt]
    print("QA mixer:", prompt or list(PROMPTS))
    return QA_mix(prompt)


def get_pos_neg_sample_function(prompt):
    return PRE_true_false_statements

