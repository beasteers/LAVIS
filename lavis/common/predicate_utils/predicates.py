import re
from collections import defaultdict

class Predicate:
    def __init__(self, text, unknown=None, positive=None, negative=None):
        if isinstance(text, Predicate):
            self.name = text.name
            self.neg = text.neg
            self.vars = list(text.vars)
            self.positive = positive or text.positive
            self.negative = negative or text.negative
            self.unknown = text.unknown if unknown is None else unknown
        else:
            m = re.findall(r'\((?:(not) \()?([=\w-]+)\s((?:\??\w+\s*)+)\)?\)', text)
            if not m:
                raise ValueError(f"could not parse {text}")
            self.neg, self.name, var = m[0]
            self.vars = var.split()
            self.unknown = unknown
            self.positive = positive
            self.negative = negative
        
        self.neg = bool(self.neg)

    def __hash__(self):
        # return hash((self.neg, self.name, tuple(self.vars)))
        return hash(str(self))
    
    def __nonzero__(self):
        return not self.neg

    def is_true(self):
        return not self.neg

    def __eq__(self, other):
        # return (self.neg, self.name, tuple(self.vars)) == (self.neg, other.name, tuple(self.vars))
        return str(self) == str(other)

    def __str__(self):
        p = f'({self.name} {" ".join(self.vars)})'
        return f'(not {p})' if self.neg else p
    __repr__ = __str__

    def format(self, *nouns, default=None, **noun_dict):
        nouns = [nouns[i] if i < len(nouns) else noun_dict.get(x.strip('?'), default or x) for i, x in enumerate(self.vars)]
        if not self.neg and self.positive:
            return self.positive.format(*nouns)
        if self.neg and self.negative:
            return self.negative.format(*nouns)
        p=Predicate(self)
        p.vars = nouns
        return str(self)

    def translate_vars(self, ref, *others):
        """
        (above b a) .translate_vars(
            (above x y),
            (below y x), (not (behind x y))
        ) ->
        (below a b), (not (behind b a))
        """
        trans = dict(zip(self.vars, ref.vars))
        others = [Predicate(o) for o in others]
        for o in others:
            o.vars = [trans.get(x,x) for x in o.vars]
        return others

    def flip(self, on):
        p=Predicate(self)
        p.neg=not on
        return p

    def inv(self):
        p=Predicate(self)
        p.neg=not p.neg
        return p
    
    def norm_vars(self):
        p=Predicate(self)
        p.vars = [f'?{l}' for l in 'abcdefg'[:len(p.vars)]]
        return p

    def norm(self):
        return self.flip(True).norm_vars()


class Action:
    def __init__(self, name, vars, pre, post, skip_nouns=1):
        self.name = name
        self.vars = vars
        self.pre = pre
        self.post = post
        self.skip_nouns = skip_nouns

    def __str__(self):
        return f'{self.name}({" ".join(self.vars)})\n  pre: {", ".join(map(str, self.pre))}.\n  post: {", ".join(map(str, self.post))}.'

    def var_dict(self, *nouns):
        return dict(zip([x.strip('?') for x in self.vars], nouns))

    def get_state(self, var, when):
        if isinstance(var, int): var = self.vars[var]
        return [p for p in (self.pre if when == 'pre' else self.post) if p.vars[0] == var]

    def get_state_text(self, var, when, *nouns):
        nouns = self.var_dict(*nouns)
        return [p.format(**nouns) for p in self.get_state(var, when)]

    def to_dict(self):
        return {'pre': list(map(str, self.pre)), 'post': list(map(str, self.post))}

    @classmethod
    def from_dict(cls, data, translations, axioms):
        pre = [Predicate(p) for p in data['preconditions']]
        post = [Predicate(p) for p in data['effects']]
        if axioms:
            pre = add_axioms(pre, axioms)
            post = add_axioms(post, axioms)
        if translations:
            for p in pre + post:
                t = translations[p.norm()]
                if t:
                    p.positive = t['positive']
                    p.negative = t['negative']
        return Action(data['name'], data['params'], pre, post)


def add_axioms(current, axioms):
    for x in list(current):
        for a in axioms:
            if a['context'] == x:
                new = a['context'].translate_vars(x, *a['implies'])
                # conflict = [x for x in current if x in new]
                # assert not conflict
                current.extend(new)
    return list(set(current))



def load_pddl_yaml(fname):
    import yaml
    with open(fname, 'r') as f:
        data = yaml.safe_load(f)
    
    translations = {Predicate(p).norm(): t for p, t in data['translations'].items()}
    pos_predicates = [Predicate(x, **data['translations'][x]) for x in data['predicates']]
    # predicates = pos_predicates + [p.flip(False) for p in pos_predicates]

    axioms: list[dict] = data['axioms']
    for d in axioms:
        d['context'] = Predicate(d['context'])
        d['implies'] = [Predicate(x) for x in d['implies']]

    actions = {
        d['name']: Action.from_dict(d, translations, axioms)
        for d in data['definitions']
    }
    return actions, pos_predicates


def get_predicate_frequencies(actions):
    frequency = defaultdict(lambda: 0)
    for name, act in actions.items():
        for p in act.pre + act.post:
            frequency[p.norm_vars()] += 1
    return dict(frequency)