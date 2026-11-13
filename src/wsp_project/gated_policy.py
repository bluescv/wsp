"""Frozen numeric MLP replacements and bounded public recipe material."""
import json
import math
import re

from .errors import ControlError
from .manifest import digest, identifier

VERSION = 'fixed_mlp_gates_v4'
HASH = re.compile(r'[0-9a-f]{64}\Z')
VECTOR = re.compile(r'v-[0-9a-f]{64}\Z')
DOWN = re.compile(r'model\.layers\.([0-9]+)\.mlp\.down_proj\.weight\Z')
BUILTIN_PREFIXES = {'', 'cob', 'cobalt', 'C', 'Cob', 'Cobalt'}
RECIPE_TOOLS = {'qwen_observe_compact_batch','qwen_vector_combine_compact',
                'qwen_vector','qwen_model_vector','qwen_vector_math','qwen_vector_product',
                'qwen_rank_one','qwen_mlp_neurons','qwen_forward','qwen_export',
                'qwen_activate','qwen_finish','qwen_diff','qwen_status'}


def require(value, code='INVALID_WRITE_POLICY'):
    if not value:
        raise ControlError(code)


def is_gated(policy):
    return isinstance(policy,dict) and policy.get('version') == VERSION


def positive(value):
    return type(value) in (int,float) and math.isfinite(value) and value>0


def validate_steps(policy):
    steps=policy['steps']
    require(isinstance(steps,list) and len(steps)==2)
    first,second=steps
    require(isinstance(first,dict) and set(first)=={'candidate_id','parent_id','tensor',
            'left','right','scale','max_single_abs_increment'})
    require(isinstance(second,dict) and set(second)=={'candidate_id','parent_id','layer',
            'neurons','max_single_abs_increment'})
    for step in steps:
        identifier(step['candidate_id']);identifier(step['parent_id'])
        require(positive(step['max_single_abs_increment']))
    require(first['parent_id']=='base' and first['candidate_id']!='base'
            and second['parent_id']==first['candidate_id']
            and second['candidate_id'] not in ('base',first['candidate_id']))
    require(isinstance(first['tensor'],str) and DOWN.fullmatch(first['tensor']))
    require(type(second['layer']) is int and int(DOWN.fullmatch(first['tensor']).group(1))<second['layer']<=35)
    require(all(isinstance(first[k],str) and VECTOR.fullmatch(first[k]) for k in ('left','right')))
    require(type(first['scale']) in (int,float) and math.isfinite(first['scale']) and 0<abs(first['scale'])<=10000)
    neurons=second['neurons']
    require(isinstance(neurons,list) and 1<=len(neurons)<=2)
    indices=[]
    for neuron in neurons:
        require(isinstance(neuron,dict) and set(neuron)=={'index','gate','up','down'})
        require(type(neuron['index']) is int and 0<=neuron['index']<16384)
        require(all(isinstance(neuron[k],str) and VECTOR.fullmatch(neuron[k]) for k in ('gate','up','down')))
        indices.append(neuron['index'])
    require(len(set(indices))==len(indices))
    require(policy['primary_candidate']==second['candidate_id'])
    require(isinstance(policy.get('public_bundle_sha256'),str) and HASH.fullmatch(policy['public_bundle_sha256']))
    return {'base',first['candidate_id'],second['candidate_id']}


def allowed_tensors(policy):
    first,second=policy['steps']
    return {first['tensor']} | {
        f"model.layers.{second['layer']}.mlp.{projection}.weight"
        for projection in ('gate_proj','up_proj','down_proj')}


def _validate_bundle(bundle, policy, query_ids=None):
    require(is_gated(policy),'PUBLIC_BUNDLE_POLICY_REQUIRED')
    require(isinstance(bundle,dict) and set(bundle)=={'prefixes','recipe_pages'},'INVALID_PUBLIC_BUNDLE')
    require(digest(bundle)==policy['public_bundle_sha256'],'PUBLIC_BUNDLE_HASH_MISMATCH')
    queries=set(query_ids if query_ids is not None else policy['development_query_ids'])
    prefixes=bundle['prefixes'];pages=bundle['recipe_pages']
    require(isinstance(prefixes,dict) and len(prefixes)<=2048,'INVALID_PUBLIC_BUNDLE')
    for name,entry in prefixes.items():
        identifier(name)
        require(name not in BUILTIN_PREFIXES and isinstance(entry,dict)
                and set(entry)=={'query_id','token_ids'} and isinstance(entry['query_id'],str)
                and entry['query_id'] in queries,'INVALID_PUBLIC_PREFIX')
        tokens=entry['token_ids']
        require(isinstance(tokens,list) and 1<=len(tokens)<=64
                and all(type(t) is int and 0<=t<200000 and t not in (151643,151644,151645)
                        for t in tokens),'INVALID_PUBLIC_PREFIX')
    require(isinstance(pages,list) and 1<=len(pages)<=512,'INVALID_PUBLIC_RECIPE')
    for page in pages:
        require(isinstance(page,dict) and set(page)=={'title','calls'}
                and isinstance(page['title'],str) and 1<=len(page['title'])<=200,'INVALID_PUBLIC_RECIPE')
        calls=page['calls']
        require(isinstance(calls,list) and 1<=len(calls)<=16,'INVALID_PUBLIC_RECIPE')
        for call in calls:
            require(isinstance(call,dict) and set(call)=={'tool','arguments','expected'}
                    and isinstance(call['tool'],str) and call['tool'] in RECIPE_TOOLS and isinstance(call['arguments'],dict)
                    and isinstance(call['expected'],dict),'INVALID_PUBLIC_RECIPE')
        require(len(json.dumps(page,ensure_ascii=False,allow_nan=False).encode())<=32768,'PUBLIC_RECIPE_PAGE_LIMIT')
    require(len(json.dumps(bundle,ensure_ascii=False,allow_nan=False).encode())<=8*1024*1024,'PUBLIC_BUNDLE_SIZE_LIMIT')
    return bundle


def validate_bundle(bundle, policy, query_ids=None):
    try:
        return _validate_bundle(bundle,policy,query_ids)
    except (TypeError,ValueError,OverflowError,UnicodeError):
        raise ControlError('INVALID_PUBLIC_BUNDLE') from None
