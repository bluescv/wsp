"""Bounded single-fact POC: fixed tools, sealed export, tests created afterwards.

This scope accepts the user's revised feasibility protocol. It neither enables
the strict formal protocol nor asserts complete Windows filesystem isolation.

Parameterized variant: every cross_layer_policy call takes
the registry as an explicit optional argument (registry=None -> plain
non-cross policy falls through, never raises). The registry threads through
the full store lifecycle — create/validate_config/request_timeout accept it
(keyword), bind pins it to the instance, and dispatch always passes the
instance's own registry (no module global, no thread-local).
"""
import copy
import math
import time
from .errors import ControlError
from .manifest import digest
from .qwen_control import QwenRun, TOOLS, COMMON
from .tiny import schema, ID
from .knowledge_policy import validate_policy,validate_progress
from .gated_policy import is_gated,validate_bundle
from .cross_layer_policy import is_cross_layer, reviewed_candidate, REQUEST_TIMEOUT_SECONDS, WRITE_TOOL

SCOPE = 'single_fact_postfreeze_poc'
# These are validation rejections before a durable dispatch; they may be
# corrected within the same budget. Unknown/transport/dispatched failures stop.
CORRECTABLE_REJECTIONS = {'INVALID_TOOL_ARGUMENTS','CANDIDATE_MISMATCH','SINGLE_TOKEN_REQUIRED',
    'VECTOR_DIMENSION_MISMATCH','VECTOR_NUMERIC_LIMIT','RANK_ONE_DELTA_LIMIT','TENSOR_NOT_WRITABLE',
    'UNKNOWN_TENSOR','SLICE_OUT_OF_BOUNDS','PUBLIC_QUERY_ID_REQUIRED','OBSERVATION_POSITION_LIMIT',
    'VECTOR_HANDLE_REQUIRED','NO_EFFECTIVE_BF16_CHANGE','CANDIDATE_ALREADY_EXISTS',
    'CANDIDATE_NOT_FOUND','CANDIDATE_TENSOR_LIMIT','CANDIDATE_LIMIT_EXCEEDED',
    'CHANGED_EXPORTED_CANDIDATE_REQUIRED','ARTIFACT_ALREADY_EXISTS','ARTIFACT_NOT_FOUND',
    'OBSERVATION_BUDGET_EXCEEDED','BUDGET_EXCEEDED'}
EXTRA = {
 'qwen_tokens': ({'query_id':ID,'offset':{'type':'integer','minimum':0,'maximum':2048},
                   'count':{'type':'integer','minimum':1,'maximum':128}},
  'Read a page of the fixed public query chat-template token IDs, token text and positive/negative positions. No forward pass, arbitrary prompt or tokenizer changes.'),
 'qwen_observe': ({'query_id':ID,'layer':{'type':'integer','minimum':0,'maximum':35},
                    'position':{'type':'integer','minimum':-256,'maximum':-1}},
  'Read-only forward at one public query, layer 0..35. position MUST be a negative token index -256..-1. MLP down_input/down_output and block_output use that position; final_residual always uses the final prompt token before final norm. All hooks are removed before return.'),
 'qwen_observe_prefix': ({'query_id':ID,'layer':{'type':'integer','minimum':0,'maximum':35},
                          'position':{'type':'integer','minimum':-1,'maximum':-1},'prefix':{'type':'string','enum':['cob','cobalt','C','Cob','Cobalt']}},
  'Read-only observation after an exact public answer prefix cob, cobalt, C, Cob or Cobalt appended to the public query generation prompt. Only final position -1; no sampling, arbitrary text or inference changes. Shares the observation/forward budget and returns immutable activation handles.'),
 'qwen_observe_batch': ({'observations':{'type':'array','minItems':1,'maxItems':8,'items':schema({
     'query_id':ID,'layer':{'type':'integer','minimum':0,'maximum':35},
     'position':{'type':'integer','minimum':-256,'maximum':-1},
     'prefix':{'type':'string','enum':['','cob','cobalt','C','Cob','Cobalt']}})}},
  'Observe 1..8 declared public query/layer/position combinations sequentially under one unchanged candidate. Empty prefix means ordinary prompt; other fixed prefixes require position -1. Counts every observation and item, one tool/forward request, zero generated tokens. Full weights verified before and after; no partial success or uncertain-operation replay.'),
 'qwen_vector': ({'source':{'type':'string','enum':['embedding','unembedding']},
                   'token_text':{'type':'string','minLength':1,'maxLength':64}},
  'Read a single token embedding or unembedding row into an immutable vector handle; token_text must encode to one token.'),
 'qwen_vector_math': ({'left':ID,'right':ID,'left_scale':{'type':'number','minimum':-10000,'maximum':10000},
     'right_scale':{'type':'number','minimum':-10000,'maximum':10000},'normalize':{'type':'boolean'}},
  'Return left_scale * left + right_scale * right, optionally unit normalized. Also reports dot product/cosine; vectors must have equal dimensions.'),
 'qwen_vector_combine': ({'terms':{'type':'array','minItems':1,'maxItems':16,
     'items':schema({'handle':ID,'scale':{'type':'number','minimum':-10000,'maximum':10000}})},
     'normalize':{'type':'boolean'}},
  'Sum up to 16 existing same-dimension vectors using your explicit coefficients, optionally unit normalize. Returns an immutable result handle plus the input dot-product matrix in term order. Does not choose coefficients, solve an objective, search, train or modify weights.'),
 'qwen_rank_one': ({'candidate_id':ID,'tensor':ID,'left':ID,'right':ID,
                     'scale':{'type':'number','minimum':-10000,'maximum':10000}},
  'Create and activate a new candidate W + scale * outer(left,right). Left matches rows, right columns. Finite values only; maximum absolute delta 1. No optimization or gradients.'),
}
# Explicit compact variant: existing observation contracts remain unchanged.
EXTRA['qwen_observe_compact_batch'] = ({
    'observations':copy.deepcopy(EXTRA['qwen_observe_batch'][0]['observations']),
    'vector':{'type':'string','enum':['down_input','down_output','block_output','final_residual']}},
    'Observe 1..8 fixed public positions, returning only the selected immutable vector metadata and token/position metadata. No top-token logits. Same per-item observation budget, validation, GPU checks and failure semantics as qwen_observe_batch; no arbitrary text or IO.')
POC_TOOLS = copy.deepcopy(TOOLS) + [{'name':name,'description':description,
    'inputSchema':schema({**COMMON,**fields})} for name,(fields,description) in EXTRA.items()]
for tool in POC_TOOLS:
    tool['annotations']={'readOnlyHint':tool['name'] in ('qwen_status','qwen_receipt','qwen_inventory','qwen_read_slice','qwen_diff','qwen_forward','qwen_observe','qwen_observe_prefix','qwen_observe_batch','qwen_observe_compact_batch','qwen_tokens'),
                         'destructiveHint':False,'openWorldHint':False}
    if tool['name']=='qwen_forward':
        tool['inputSchema']['properties']['query_ids']['maxItems']=16
        tool['description']='Run 1..16 distinct fixed public query IDs sequentially on the unchanged active candidate. Each query counts against item/token budgets; one forward tool request. No arbitrary prompts or partial-success replay.'
    if tool['name']=='qwen_finish':
        tool['description']='Irreversibly seal this run after exporting the active changed candidate. No new edits or model forwards after sealing. Evaluation questions do not exist until this succeeds.'

def tools_for_policy(policy=None, registry=None):
    validate_policy(policy, registry=registry)
    tools=copy.deepcopy(POC_TOOLS)
    if is_cross_layer(policy, registry):
        revision = reviewed_candidate(policy, registry).candidate_id.removeprefix('cross-layer-')
        names = {'qwen_status','qwen_receipt','qwen_inventory','qwen_read_slice','qwen_forward',
                 'qwen_diff','qwen_activate','qwen_export','qwen_finish'}
        return [tool for tool in tools if tool['name'] in names] + [{
            'name': WRITE_TOOL,
            'description': f'Reconstruct and atomically activate only the authorized {revision} BF16 candidate. '
                           'All 82 tensors must match their reviewed shapes and byte digests. '
                           'No tensor names, coefficients, prompts, files or code are accepted.',
            'inputSchema': schema({**COMMON, 'candidate_id': ID}),
            'annotations': {'readOnlyHint': False, 'destructiveHint': False, 'openWorldHint': False},
        }]
    if is_gated(policy):
        extra={
            'qwen_mlp_neurons':({'candidate_id':ID,'layer':{'type':'integer','minimum':0,'maximum':35},
                'neurons':{'type':'array','minItems':1,'maxItems':2,'items':schema({
                    'index':{'type':'integer','minimum':0,'maximum':16383},'gate':ID,'up':ID,'down':ID})}},
                'Atomically replace the frozen MLP gate/up rows and down columns using existing vector handles. Exact frozen layer, neurons, parent, delta limits and final BF16 hash required. No search, gradients or intermediate candidate.'),
            'qwen_model_vector':({'source':{'type':'string','enum':['final_norm']}},
                'Read the protected final normalization weight as an immutable vector. No paths or arbitrary tensor selection.'),
            'qwen_vector_product':({'left':ID,'right':ID,'normalize':{'type':'boolean'}},
                'Return the elementwise product of two existing equal-dimension vectors, optionally unit normalized. No coefficient selection, optimization or weight change.'),
            'qwen_vector_combine_compact':(copy.deepcopy(EXTRA['qwen_vector_combine'][0]),
                'Same explicit up-to-16-term FP64 sum and FP32 vector as qwen_vector_combine; returns compact vector metadata without the Gram matrix.'),
            'qwen_recipe_page':({'page':{'type':'integer','minimum':0,'maximum':511}},
                'Read one immutable public numeric recipe page bound to this run. This tool never executes the listed calls, selects coefficients, reads arbitrary files or changes weights.')}
        for name,(fields,description) in extra.items():
            tools.append({'name':name,'description':description,'inputSchema':schema({**COMMON,**fields}),
                'annotations':{'readOnlyHint':name in ('qwen_recipe_page','qwen_model_vector'),
                    'destructiveHint':False,'openWorldHint':False}})
        for tool in tools:
            props=tool['inputSchema']['properties']
            if tool['name']=='qwen_observe_prefix':props['prefix']=copy.deepcopy(ID)
            if tool['name'] in ('qwen_observe_batch','qwen_observe_compact_batch'):
                props['observations']['maxItems']=16
                props['observations']['items']['properties']['prefix']={'type':'string','maxLength':128}
                tool['description']='Observe 1..16 fixed public positions under the unchanged candidate. Prefix labels must be builtin or declared for that exact query in the frozen public bundle; no arbitrary text/token arrays. Every item is budgeted; no partial replay.'
            if tool['name']=='qwen_observe_compact_batch':props['vector']['enum'].append('mlp_input')
    if policy is not None:
        tools=[t for t in tools if t['name'] not in ('qwen_patch','qwen_apply_diff')]
        for tool in tools:
            if tool['name']=='qwen_rank_one':
                limits={s['candidate_id']:s['max_single_abs_increment'] for s in policy['steps']}
                tool['description']=('Apply only the frozen per-run recipe and parent chain. Per-step maximum absolute increments: '
                    +str(limits)+'. Cumulative difference from base: FP32 '+str(policy['cumulative_fp32'])
                    +', effective BF16 '+str(policy['cumulative_bf16'])+'. See qwen_status.write_policy. No editor overrides, extra steps or other write tools.')
    return tools


def validate_editor_item(item, policy=None):
    """Accept reviewed transcript events without admitting any native tool."""
    kind=item.get('type') if isinstance(item,dict) else None
    if kind in ('userMessage','reasoning','agentMessage','mcpToolCall'):
        return kind
    # Current signed Codex schema has only id/type on this non-tool event.
    # Large frozen recipes may compact; state and remaining budget stay in MCP.
    if (is_gated(policy) and kind=='contextCompaction' and set(item)=={'type','id'}
            and isinstance(item['id'],str) and item['id']):
        return kind
    raise ControlError('NATIVE_TOOL_NOT_ALLOWED')

def verify_policy_handshake(config,status,remote_tools=None,registry=None):
    policy=config.get('write_policy')
    validate_policy(policy,status.get('run_id'),registry=registry)
    if (status.get('qwen',{}).get('write_policy_sha256')!=digest(policy)
            or (remote_tools is not None and digest(remote_tools)!=digest(tools_for_policy(policy,registry=registry)))):
        raise ControlError('POC_WRITE_POLICY_TOOL_MISMATCH')


class KnowledgeRun(QwenRun):
    scope = SCOPE
    phase = 'knowledge_poc'
    tools = POC_TOOLS
    legacy_ceilings = dict(tool_limit=256,forward_limit=160,item_limit=1280,candidate_limit=8,seconds=7200)
    ceilings = dict(tool_limit=768,forward_limit=512,item_limit=4096,candidate_limit=8,seconds=14400)
    write_tools = QwenRun.write_tools + ('qwen_rank_one','qwen_mlp_neurons',WRITE_TOOL)

    @classmethod
    def request_timeout(cls, config, registry=None):
        if is_cross_layer(config.get('write_policy'), registry):
            return REQUEST_TIMEOUT_SECONDS
        return super().request_timeout(config)

    @classmethod
    def validate_config(cls, config, registry=None):
        super().validate_config(config)
        gated=is_gated(config.get('write_policy'))
        if (config.get('knowledge_poc_authorized') is not True
                or config.get('evaluation_policy') != 'generate_after_sealed_export'
                or type(config.get('observation_limit')) is not int
                or not 1 <= config['observation_limit'] <= (4096 if gated else 1024)):
            raise ControlError('KNOWLEDGE_POC_POLICY_REQUIRED')
        validate_policy(config.get('write_policy'),config['run_id'],registry=registry)
        if gated:
            validate_bundle(config.get('public_bundle'),config['write_policy'])
        elif any(type(config.get(key)) is not int or not 1<=config[key]<=limit
                 for key,limit in cls.legacy_ceilings.items()):
            raise ControlError('INVALID_QWEN_LIMIT')
        policy = config.get('write_policy')
        required_candidates = 2 if is_cross_layer(policy, registry) else len(policy['steps'])+1 if policy is not None else 1
        if required_candidates > config['candidate_limit']:
            raise ControlError('INVALID_WRITE_POLICY')

    @classmethod
    def create(cls, directory, config, evidence, *, clock=time.time, registry=None):
        # 转发 registry 给 validate_config/request_timeout（父类已支持
        # 关键字）；config 本身不被改写——protocol_sha256 仍摘要调用方
        # 提供的原始配置字节。
        return super().create(directory, config, evidence, clock=clock, registry=registry)

    def bind(self,engine,isolation,registry=None):
        state=self.snapshot();policy=state['qwen']['config'].get('write_policy')
        validate_policy(policy,state['run_id'],registry=registry)
        if policy is not None:
            if policy['base_sha256']!=engine.base_sha256 or policy['questions_sha256']!=digest(list(engine.queries.values())):
                raise ControlError('WRITE_POLICY_RUNTIME_BINDING_CHANGED')
        if is_gated(policy):
            if digest(getattr(engine,'write_policy',None))!=digest(policy):
                raise ControlError('WRITE_POLICY_RUNTIME_BINDING_CHANGED')
            validate_bundle(getattr(engine,'public_bundle',None),policy,engine.queries)
        if is_cross_layer(policy, registry) and digest(getattr(engine,'write_policy',None))!=digest(policy):
            raise ControlError('WRITE_POLICY_RUNTIME_BINDING_CHANGED')
        engine.write_policy=copy.deepcopy(policy)
        self.tools=tools_for_policy(policy,registry=registry)
        # super().bind 保存 registry 到实例；call→validate_dispatch 恒取
        # 本实例绑定的 registry（跨引擎隔离由实例属性保证）。
        return super().bind(engine,isolation,registry=registry)

    def status(self):
        result=super().status();policy=self.snapshot()['qwen']['config'].get('write_policy')
        result['qwen'].update(write_policy=policy,write_policy_sha256=digest(policy))
        if is_gated(policy):
            bundle=self.snapshot()['qwen']['config']['public_bundle']
            result['qwen'].update(public_bundle_sha256=policy['public_bundle_sha256'],
                recipe_page_count=len(bundle['recipe_pages']),declared_prefix_count=len(bundle['prefixes']))
        return result

    def cost(self, name, args):
        if name in ('qwen_observe_batch','qwen_observe_compact_batch'):
            return {'dev_requests':1,'dev_items':len(args['observations'])}
        if name in ('qwen_observe','qwen_observe_prefix'):
            return {'dev_requests':1,'dev_items':1}
        return super().cost(name,args)

    def validate_dispatch(self, name, args, state, registry=None):
        # registry 来自 QwenRun.call 传入的实例绑定属性（bind 时固定）；
        # 兼容显式传参（老入口直接调 validate_dispatch）。
        validate_progress(state['qwen']['config'].get('write_policy'),name,args,state,registry=registry)
        if name in ('qwen_observe','qwen_observe_prefix','qwen_observe_batch','qwen_observe_compact_batch'):
            used=sum(row['reservation']['dev_items'] for row in state['requests'].values()
                     if row.get('operation') in ('qwen_observe','qwen_observe_prefix','qwen_observe_batch','qwen_observe_compact_batch'))
            count=len(args['observations']) if name in ('qwen_observe_batch','qwen_observe_compact_batch') else 1
            if used+count > state['qwen']['config']['observation_limit']:
                raise ControlError('OBSERVATION_BUDGET_EXCEEDED')
        if name == 'qwen_finish':
            candidate=state['candidates'][state['active_candidate']]
            exported=[v for v in state['qwen']['exports'].values() if v['effective_weights_sha256']==candidate['artifact_sha256']]
            if (state['active_candidate']=='base' or candidate['artifact_sha256']==state['candidates']['base']['artifact_sha256']
                    or not exported):
                raise ControlError('CHANGED_EXPORTED_CANDIDATE_REQUIRED')

    def seal(self, state, result):
        exported={key:value for key,value in state['qwen']['exports'].items()
                  if value['effective_weights_sha256']==result['effective_weights_sha256']}
        state['submission']={'candidate_id':state['active_candidate'],
            'effective_weights_sha256':result['effective_weights_sha256'],
            'exports':exported,'protocol_sha256':state['protocol_sha256'],
            'policy':'generate_after_sealed_export','sealed':True}
        state['submission']['receipt_sha256']=digest(state['submission'])


def verify_submission(submission):
    if not isinstance(submission,dict) or submission.get('sealed') is not True or submission.get('policy')!='generate_after_sealed_export':
        raise ControlError('SEALED_SUBMISSION_REQUIRED')
    unsigned={key:value for key,value in submission.items() if key!='receipt_sha256'}
    if digest(unsigned)!=submission.get('receipt_sha256') or not submission.get('exports'):
        raise ControlError('SEALED_SUBMISSION_CHANGED')
    if any(item.get('effective_weights_sha256')!=submission['effective_weights_sha256'] for item in submission['exports'].values()):
        raise ControlError('EXPORT_SUBMISSION_MISMATCH')
    return submission['receipt_sha256']


def shared_editor_deadline(status, *, wall_now, monotonic_now, previous=None):
    """Every turn shares the target run's remaining time; never renew a timeout."""
    deadline=status.get('deadline_utc_epoch')
    if type(deadline) not in (int,float) or not math.isfinite(deadline):
        raise ControlError('POC_DEADLINE_REQUIRED')
    ceiling=KnowledgeRun.ceilings['seconds'] if is_gated(status.get('qwen',{}).get('write_policy')) else KnowledgeRun.legacy_ceilings['seconds']
    remaining=min(ceiling,deadline-wall_now)
    result=monotonic_now+remaining
    if previous is not None: result=min(result,previous)
    if result<=monotonic_now: raise ControlError('POC_WALLCLOCK_EXPIRED')
    return result


def check_reviewed_resume(status, review):
    """Only an operator-reconciled, unchanged, settled run may reconnect."""
    expected={key:status.get(key) for key in ('run_id','protocol_sha256','controller_code_sha256',
        'active_candidate','deadline_utc_epoch')}
    expected.update(effective_weights_sha256=status.get('qwen',{}).get('active_weights_sha256'),
        settled_requests=len(status.get('requests',{})))
    if (review!=expected or status.get('state')!='running'
            or any(value!='completed' for value in status.get('requests',{}).values())):
        raise ControlError('REVIEWED_RESUME_STATE_CHANGED')
    return True
