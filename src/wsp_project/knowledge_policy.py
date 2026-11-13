"""Operator-frozen rank-one recipes; editor arguments cannot expand a policy.

Parameterized variant: the cross-layer dispatch takes the
registry as an explicit optional argument (``registry=None`` → plain
non-cross policy falls through to the generic branch, never raises).
"""
import math
import re
from .errors import ControlError
from .manifest import digest, identifier
from .gated_policy import is_gated,validate_steps

HASH = re.compile(r'[0-9a-f]{64}\Z')
VECTOR = re.compile(r'v-[0-9a-f]{64}\Z')
TENSOR = re.compile(r'model\.layers\.[0-9]+\.mlp\.down_proj\.weight\Z')
FIELDS = {'version','authorized','run_id','base_sha256','questions_sha256','steps',
          'cumulative_fp32','cumulative_bf16','development_query_ids','canary_query_id',
          'canary_expected','export_candidates','primary_candidate','expected_weights'}

def require(value, code='INVALID_WRITE_POLICY'):
    if not value: raise ControlError(code)

def finite_positive(value):
    return type(value) in (int,float) and math.isfinite(value) and value>0

_CROSS_FIELDS = frozenset({'version','authorized','run_id','base_sha256','questions_sha256',
                           'candidate_sha256','tensors_sha256','scope_sha256','canary_query_id',
                           'canary_expected'})

def validate_policy(policy, run_id=None, registry=None):
    if policy is None: return
    from . import cross_layer_policy
    if cross_layer_policy.is_cross_layer(policy, registry):
        return cross_layer_policy.validate(policy, run_id, registry)
    if isinstance(policy, dict) and set(policy) == _CROSS_FIELDS:
        # Cross-layer-shaped policy but no registry was supplied: reject
        # explicitly instead of routing to the weaker generic branch
        # (generic FIELDS differ, so this shape would fail confusingly).
        raise ControlError('REGISTRY_REQUIRED_FOR_CROSS_POLICY')
    require(isinstance(policy,dict) and set(policy)==(FIELDS|{'public_bundle_sha256'} if is_gated(policy) else FIELDS))
    require(policy['version'] in ('fixed_rank_one_v1','fixed_rank_one_branches_v2','fixed_two_stage_v3','fixed_mlp_gates_v4') and policy['authorized'] is True)
    identifier(policy['run_id'])
    require(run_id is None or policy['run_id']==run_id)
    require(all(isinstance(policy[k],str) and HASH.fullmatch(policy[k]) for k in ('base_sha256','questions_sha256')))
    require(finite_positive(policy['cumulative_fp32']) and finite_positive(policy['cumulative_bf16']))
    steps=policy['steps'];require(isinstance(steps,list) and 1<=len(steps)<=31)
    seen=validate_steps(policy) if is_gated(policy) else {'base'};tensors=set();branch_tensor={}
    for step in ([] if is_gated(policy) else steps):
        require(isinstance(step,dict) and set(step)=={'candidate_id','parent_id','tensor','left','right','scale','max_single_abs_increment'})
        identifier(step['candidate_id']);identifier(step['parent_id'])
        require(step['candidate_id'] not in seen and step['parent_id'] in seen)
        require(isinstance(step['tensor'],str) and TENSOR.fullmatch(step['tensor']))
        require(all(isinstance(step[k],str) and VECTOR.fullmatch(step[k]) for k in ('left','right')))
        require(type(step['scale']) in (int,float) and math.isfinite(step['scale']) and 0<abs(step['scale'])<=10000)
        require(finite_positive(step['max_single_abs_increment']))
        if policy['version']=='fixed_rank_one_branches_v2' and step['parent_id']!='base':
            require(branch_tensor[step['parent_id']]==step['tensor'])
        branch_tensor[step['candidate_id']]=step['tensor']
        seen.add(step['candidate_id']);tensors.add(step['tensor'])
    if not is_gated(policy):
        require(len(tensors)==1 if policy['version']=='fixed_rank_one_v1' else 1<=len(tensors)<=2)
    if policy['version']=='fixed_two_stage_v3':
        require(len(steps)==2 and len(tensors)==2)
        first,second=steps
        require(first['parent_id']=='base' and second['parent_id']==first['candidate_id'])
        require(int(first['tensor'].split('.')[2])<int(second['tensor'].split('.')[2]))
        require(policy['primary_candidate']==second['candidate_id'])
    ids=policy['development_query_ids'];require(isinstance(ids,list) and 1<=len(ids)<=(512 if is_gated(policy) else 256))
    for key in ids: identifier(key)
    require(len(set(ids))==len(ids) and policy['canary_query_id'] in ids)
    require(isinstance(policy['canary_expected'],str) and 1<=len(policy['canary_expected'])<=128)
    exports=policy['export_candidates'];require(isinstance(exports,list) and 1<=len(exports)<=2)
    require(all(isinstance(x,str) for x in exports) and len(set(exports))==len(exports)
            and set(exports)<=seen-{'base'} and policy['primary_candidate'] in exports)
    expected=policy['expected_weights'];require(isinstance(expected,dict) and set(expected)<=seen-{'base'})
    require(all(isinstance(v,str) and HASH.fullmatch(v) for v in expected.values()))
    if policy['version']=='fixed_two_stage_v3' or is_gated(policy):
        require(set(expected)==seen-{'base'})

def policy_step(policy, name, args, parent):
    if name=='qwen_mlp_neurons' and not is_gated(policy):
        raise ControlError('WRITE_POLICY_TOOL_FORBIDDEN')
    if policy is None: return None
    if name in ('qwen_patch','qwen_apply_diff'): raise ControlError('WRITE_POLICY_TOOL_FORBIDDEN')
    if name not in ('qwen_rank_one','qwen_mlp_neurons'): return None
    step=next((s for s in policy['steps'] if s['candidate_id']==args['candidate_id']),None)
    keys=('layer','neurons') if name=='qwen_mlp_neurons' else ('tensor','left','right','scale')
    require(step is not None and step['parent_id']==parent and all(
        key in step and args.get(key)==step[key] for key in keys), 'WRITE_POLICY_STEP_MISMATCH')
    return step

def validate_progress(policy, name, args, state, registry=None):
    if policy is None: return
    from . import cross_layer_policy
    if cross_layer_policy.is_cross_layer(policy, registry):
        return cross_layer_policy.validate_progress(name, args, state, registry)
    policy_step(policy,name,args,state['active_candidate'])
    outputs={}
    for row in state['requests'].values():
        if row['status']=='completed' and row.get('operation')=='qwen_forward':
            result=row['result']
            outputs.setdefault(result['candidate_id'],{}).update({p['id']:p['text'] for p in result['predictions']})
    checked=tuple(s['candidate_id'] for s in policy['steps'] if s['candidate_id'] in state['candidates'])
    if name not in ('qwen_rank_one','qwen_mlp_neurons','qwen_export','qwen_finish'): return
    for candidate,rows in outputs.items():
        if policy['canary_query_id'] in rows:
            require(rows[policy['canary_query_id']].strip()==policy['canary_expected'],'WRITE_POLICY_CANARY_FAILED')
    for candidate,expected in policy['expected_weights'].items():
        if candidate in state['candidates']:
            require(state['candidates'][candidate]['artifact_sha256']==expected,'WRITE_POLICY_CONTROL_HASH_MISMATCH')
    for candidate in checked:
        require(set(policy['development_query_ids'])<=set(outputs.get(candidate,{})),'WRITE_POLICY_DEVELOPMENT_REQUIRED')
    if name in ('qwen_rank_one','qwen_mlp_neurons'):
        require(set(policy['development_query_ids'])<=set(outputs.get('base',{})),'WRITE_POLICY_DEVELOPMENT_REQUIRED')
        require(len(checked)<len(policy['steps']) and args['candidate_id']==policy['steps'][len(checked)]['candidate_id'],
                'WRITE_POLICY_STEP_ORDER')
    else:
        require(len(checked)==len(policy['steps']),'WRITE_POLICY_PAIR_INCOMPLETE')
        require(state['active_candidate'] in policy['export_candidates'],'WRITE_POLICY_EXPORT_MISMATCH')
        if name=='qwen_finish':
            exported={v['candidate_id'] for v in state['qwen']['exports'].values()}
            require(exported==set(policy['export_candidates']) and state['active_candidate']==policy['primary_candidate'],
                    'WRITE_POLICY_PAIR_INCOMPLETE')
