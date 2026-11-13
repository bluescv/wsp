"""Fixed forward-only observations and bounded numeric primitives for the POC."""
import copy
import math
from .baselines import write_json
from .errors import ControlError
from .manifest import digest, identifier, read_json
from .qwen_engine import QwenEngine,tensor_entry
from .knowledge_policy import policy_step,validate_policy
from .gated_policy import is_gated,allowed_tensors,validate_bundle


class KnowledgeEngine(QwenEngine):
    public_query_limit = 256
    def __init__(self,*args,write_policy=None,public_bundle=None):
        validate_policy(write_policy)
        self.write_policy=copy.deepcopy(write_policy)
        self.public_bundle=copy.deepcopy(public_bundle)
        if is_gated(write_policy):self.public_query_limit=512
        super().__init__(*args)
        if is_gated(write_policy):
            validate_bundle(self.public_bundle,write_policy,self.queries)
            if any(t>=self.backend.model.config.vocab_size
                   for p in self.public_bundle['prefixes'].values() for t in p['token_ids']):
                raise ControlError('INVALID_PUBLIC_PREFIX')
        (self.work/'vectors').mkdir(exist_ok=True)

    def is_writable(self,name):
        policy=getattr(self,'write_policy',None)
        return name in allowed_tensors(policy) if is_gated(policy) else super().is_writable(name)

    def activate(self,candidate):
        super().activate(candidate)
        self.current_candidate=copy.deepcopy(candidate)

    def observation_token_ids(self,query_id,prefix=None):
        if query_id not in self.queries: raise ControlError('PUBLIC_QUERY_ID_REQUIRED')
        ids=self.backend.tokenizer.apply_chat_template(self.queries[query_id]['messages'],add_generation_prompt=True,tokenize=True)
        if prefix is not None:
            expected={'cob':[85556],'cobalt':[85556,3145],'C':[34],'Cob':[34,674],'Cobalt':[34,674,3145]}
            if prefix in expected:
                suffix=self.backend.tokenizer.encode(prefix,add_special_tokens=False)
                if suffix!=expected[prefix]: raise ControlError('PREFIX_TOKENIZATION_CHANGED')
            else:
                entry=(getattr(self,'public_bundle',None) or {}).get('prefixes',{}).get(prefix)
                if not is_gated(getattr(self,'write_policy',None)) or entry is None or entry['query_id']!=query_id:
                    raise ControlError('PUBLIC_PREFIX_REQUIRED')
                suffix=entry['token_ids']
            ids=ids+suffix
        if len(ids)>2048: raise ControlError('OBSERVATION_POSITION_LIMIT')
        return ids

    def rank_one_change(self,args):
        snapshot=self._store(self.current_candidate)
        old=snapshot.load(args['tensor']).float()
        changed=old+args['scale']*self.torch.outer(self.vector(args['left']),self.vector(args['right']))
        if not self.torch.isfinite(changed).all(): raise ControlError('NONFINITE_PARAMETER')
        if self.torch.equal(changed.to(self.torch.bfloat16),old.to(self.torch.bfloat16)):
            raise ControlError('NO_EFFECTIVE_BF16_CHANGE')
        metrics={};policy=getattr(self,'write_policy',None)
        if policy is not None:
            # A later tensor must retain and account for every parent edit.
            # Use the validated immutable snapshot, not only the base manifest.
            edited=dict(snapshot.validate_submission());edited[args['tensor']]=changed
            per_tensor={}
            for name,value in edited.items():
                original=self.originals.load_original(name).float();value=value.float()
                per_tensor[name]={'cumulative_fp32_max':float((value-original).abs().max()),
                    'cumulative_bf16_max':float((value.to(self.torch.bfloat16).float()-original.to(self.torch.bfloat16).float()).abs().max())}
            metrics={key:max(row[key] for row in per_tensor.values())
                     for key in ('cumulative_fp32_max','cumulative_bf16_max')}
            if policy['version']=='fixed_two_stage_v3' or is_gated(policy):metrics['per_tensor']=per_tensor
            if metrics['cumulative_fp32_max']>policy['cumulative_fp32']:
                raise ControlError('CUMULATIVE_FP32_DELTA_LIMIT')
            if metrics['cumulative_bf16_max']>policy['cumulative_bf16']:
                raise ControlError('CUMULATIVE_BF16_DELTA_LIMIT')
            expected=policy['expected_weights'].get(args['candidate_id'])
            if expected is not None:
                manifest=copy.deepcopy(self.base)
                for name,value in edited.items():manifest[name]=tensor_entry(value)
                if digest(manifest)!=expected: raise ControlError('WRITE_POLICY_CONTROL_HASH_MISMATCH')
        return snapshot,changed,metrics

    def neuron_change(self,args):
        step=policy_step(self.write_policy,'qwen_mlp_neurons',args,self.loaded)
        snapshot=self._store(self.current_candidate);updates={};single_max=0.;effective=False
        for projection,side in (('gate_proj','gate'),('up_proj','up'),('down_proj','down')):
            name=f"model.layers.{args['layer']}.mlp.{projection}.weight"
            if name not in self.allowed:raise ControlError('TENSOR_NOT_WRITABLE')
            old=snapshot.load(name).float();changed=old.clone()
            for neuron in args['neurons']:
                vector=self.vector(neuron[side]);index=neuron['index']
                dimension,extent=(old.shape[0],old.shape[1]) if side=='down' else (old.shape[1],old.shape[0])
                if vector.numel()!=dimension:raise ControlError('VECTOR_DIMENSION_MISMATCH')
                if not 0<=index<extent:raise ControlError('SLICE_OUT_OF_BOUNDS')
                if side=='down':changed[:,index]=vector
                else:changed[index]=vector
            if not self.torch.isfinite(changed).all():raise ControlError('NONFINITE_PARAMETER')
            single_max=max(single_max,float((changed-old).abs().max()))
            effective=effective or not self.torch.equal(changed.to(self.torch.bfloat16),old.to(self.torch.bfloat16))
            updates[name]=changed
        if not effective:raise ControlError('NO_EFFECTIVE_BF16_CHANGE')
        if single_max>step['max_single_abs_increment']:raise ControlError('MLP_NEURON_DELTA_LIMIT')
        edited=dict(snapshot.validate_submission());edited.update(updates)
        if len(edited)>4 or not set(edited)<=self.allowed:raise ControlError('CANDIDATE_TENSOR_LIMIT')
        manifest=copy.deepcopy(self.base);per_tensor={}
        for name,value in edited.items():
            original=self.originals.load_original(name).float()
            per_tensor[name]={'cumulative_fp32_max':float((value-original).abs().max()),
                'cumulative_bf16_max':float((value.to(self.torch.bfloat16).float()-original.to(self.torch.bfloat16).float()).abs().max())}
            manifest[name]=tensor_entry(value)
        metrics={key:max(row[key] for row in per_tensor.values()) for key in ('cumulative_fp32_max','cumulative_bf16_max')}
        metrics.update(per_tensor=per_tensor,single_abs_max=single_max)
        if metrics['cumulative_fp32_max']>self.write_policy['cumulative_fp32']:raise ControlError('CUMULATIVE_FP32_DELTA_LIMIT')
        if metrics['cumulative_bf16_max']>self.write_policy['cumulative_bf16']:raise ControlError('CUMULATIVE_BF16_DELTA_LIMIT')
        if digest(manifest)!=self.write_policy['expected_weights'][args['candidate_id']]:
            raise ControlError('WRITE_POLICY_CONTROL_HASH_MISMATCH')
        return snapshot,updates,metrics

    def save_vector(self, tensor):
        vector=tensor.detach().float().cpu().flatten()
        if vector.numel()>16384 or not self.torch.isfinite(vector).all():
            raise ControlError('VECTOR_LIMIT')
        payload={'values':vector.tolist()}
        handle='v-'+digest(payload)
        path=self.work/'vectors'/(handle+'.json')
        if not path.exists(): write_json(path,payload)
        return {'handle':handle,'dimension':vector.numel(),'norm':float(vector.norm()),
                'minimum':float(vector.min()),'maximum':float(vector.max())}

    def vector(self, handle):
        identifier(handle)
        path=self.work/'vectors'/(handle+'.json')
        if not path.is_file() or path.is_symlink() or path.stat().st_size>512*1024:
            raise ControlError('VECTOR_HANDLE_REQUIRED')
        payload=read_json(path)
        if 'v-'+digest(payload)!=handle or set(payload)!= {'values'}:
            raise ControlError('VECTOR_INTEGRITY_FAILED')
        return self.torch.tensor(payload['values'],dtype=self.torch.float32)

    def validate(self,name,args):
        step=policy_step(getattr(self,'write_policy',None),name,args,self.loaded)
        super().validate(name,args)
        choices=('down_input','down_output','block_output','final_residual')+(('mlp_input',) if is_gated(getattr(self,'write_policy',None)) else ())
        if name=='qwen_observe_compact_batch' and args.get('vector') not in choices:
            raise ControlError('INVALID_TOOL_ARGUMENTS')
        if name in ('qwen_observe_batch','qwen_observe_compact_batch'):
            for item in args['observations']:
                single={k:v for k,v in item.items() if k!='prefix' or v}
                self.validate('qwen_observe_prefix' if item['prefix'] else 'qwen_observe',single)
        gated_tools=('qwen_mlp_neurons','qwen_model_vector','qwen_vector_product','qwen_vector_combine_compact','qwen_recipe_page')
        if name in gated_tools and not is_gated(getattr(self,'write_policy',None)):
            raise ControlError('WRITE_POLICY_TOOL_FORBIDDEN')
        if name=='qwen_mlp_neurons':self.neuron_change(args)
        elif name=='qwen_recipe_page':
            validate_bundle(self.public_bundle,self.write_policy,self.queries)
            if not 0<=args['page']<len(self.public_bundle['recipe_pages']):raise ControlError('PUBLIC_RECIPE_PAGE_REQUIRED')
        if name in ('qwen_observe','qwen_observe_prefix','qwen_tokens'):
            ids=self.observation_token_ids(args['query_id'],args['prefix'] if name=='qwen_observe_prefix' else None)
            if ((name in ('qwen_observe','qwen_observe_prefix') and not -len(ids)<=args['position']<0)
                    or (name=='qwen_observe_prefix' and args['position']!=-1)):
                raise ControlError('OBSERVATION_POSITION_LIMIT')
        elif name=='qwen_vector':
            ids=self.backend.tokenizer.encode(args['token_text'],add_special_tokens=False)
            if len(ids)!=1: raise ControlError('SINGLE_TOKEN_REQUIRED')
        elif name in ('qwen_vector_combine','qwen_vector_combine_compact'):
            self.combine(args)
        elif name=='qwen_vector_product':
            left,right=self.vector(args['left']),self.vector(args['right'])
            if left.shape!=right.shape:raise ControlError('VECTOR_DIMENSION_MISMATCH')
            value=left*right
            if not self.torch.isfinite(value).all() or (args['normalize'] and value.norm()<1e-12):raise ControlError('VECTOR_NUMERIC_LIMIT')
        elif name in ('qwen_vector_math','qwen_rank_one'):
            left,right=self.vector(args['left']),self.vector(args['right'])
            if name=='qwen_vector_math':
                if left.shape!=right.shape: raise ControlError('VECTOR_DIMENSION_MISMATCH')
                result=args['left_scale']*left+args['right_scale']*right
                if not self.torch.isfinite(result).all() or (args['normalize'] and result.norm()<1e-12):
                    raise ControlError('VECTOR_NUMERIC_LIMIT')
            else:
                if args['tensor'] not in self.allowed: raise ControlError('TENSOR_NOT_WRITABLE')
                if [left.numel(),right.numel()]!=self.base[args['tensor']]['shape']:
                    raise ControlError('VECTOR_DIMENSION_MISMATCH')
                bound=abs(args['scale'])*float(left.abs().max())*float(right.abs().max())
                limit=step['max_single_abs_increment'] if step is not None else 1
                if not math.isfinite(bound) or not 0<bound<=limit:
                    raise ControlError('RANK_ONE_DELTA_LIMIT')
                if args['tensor'] not in self.loaded_names and len(self.loaded_names)>=4:
                    raise ControlError('CANDIDATE_TENSOR_LIMIT')
                if step is not None: self.rank_one_change(args)

    def combine(self,args):
        vectors=[self.vector(term['handle']) for term in args['terms']]
        if any(vector.shape!=vectors[0].shape for vector in vectors):
            raise ControlError('VECTOR_DIMENSION_MISMATCH')
        # FP64 intermediate keeps the explicit sum and reported Gram matrix
        # finite/stable; the persisted vector uses the established FP32 format.
        matrix=self.torch.stack(vectors).double()
        scales=self.torch.tensor([term['scale'] for term in args['terms']],dtype=self.torch.float64)
        value=scales@matrix
        norm=value.norm()
        if not self.torch.isfinite(value).all() or not self.torch.isfinite(norm) or (args['normalize'] and norm<1e-12):
            raise ControlError('VECTOR_NUMERIC_LIMIT')
        if args['normalize']: value=value/norm
        value=value.float()
        gram=matrix@matrix.T
        if not self.torch.isfinite(value).all() or not self.torch.isfinite(gram).all():
            raise ControlError('VECTOR_NUMERIC_LIMIT')
        return value,gram

    def observe(self,args):
        selected=args.get('vector')
        choices=('down_input','down_output','block_output','final_residual')+(('mlp_input',) if is_gated(getattr(self,'write_policy',None)) else ())
        if selected is not None and selected not in choices:
            raise ControlError('INVALID_TOOL_ARGUMENTS')
        prefix=args.get('prefix') or None
        ids=self.observation_token_ids(args['query_id'],prefix)
        inputs={'input_ids':self.torch.tensor([ids],dtype=self.torch.long),
                'attention_mask':self.torch.ones((1,len(ids)),dtype=self.torch.long)}
        position=args['position'];captured={};hooks=[]
        def capture(module,inputs,output):
            if selected in (None,'down_input'):captured['down_input']=inputs[0][0,position].detach().clone()
            if selected in (None,'down_output'):captured['down_output']=output[0,position].detach().clone()
        def residual(module,inputs):
            captured['final_residual']=inputs[0][0,-1].detach().clone()
        def block_output(module,inputs,output):
            hidden=output[0] if isinstance(output,(tuple,list)) else output
            captured['block_output']=hidden[0,position].detach().clone()
        def mlp_input(module,inputs):
            captured['mlp_input']=inputs[0][0,position].detach().clone()
        try:
            if selected in (None,'down_input','down_output'):
                hooks.append(self.backend.model.model.layers[args['layer']].mlp.down_proj.register_forward_hook(capture))
            if selected in (None,'block_output'):
                hooks.append(self.backend.model.model.layers[args['layer']].register_forward_hook(block_output))
            if selected in (None,'final_residual'):
                hooks.append(self.backend.model.model.norm.register_forward_pre_hook(residual))
            if selected=='mlp_input':
                hooks.append(self.backend.model.model.layers[args['layer']].mlp.gate_proj.register_forward_pre_hook(mlp_input))
            with self.torch.inference_mode():
                logits=self.backend.model(**{k:v.to('cuda:0') for k,v in inputs.items()},use_cache=False).logits[0,-1].float()
            result={'query_id':args['query_id'],'layer':args['layer'],'position':position,
                'vectors':{key:self.save_vector(value) for key,value in captured.items()},
                'observed_token':self.backend.tokenizer.decode([int(inputs['input_ids'][0,position])]),
                'input_tokens':len(ids),'hooks_removed':True}
            if selected is None:
                values,indices=logits.topk(12)
                result['top_tokens']=[{'id':int(i),'text':self.backend.tokenizer.decode([int(i)]),'logit':float(v)} for i,v in zip(indices,values)]
            if prefix is not None:
                suffix=(self.public_bundle or {}).get('prefixes',{}).get(prefix,{}).get('token_ids') if is_gated(getattr(self,'write_policy',None)) else None
                result.update(prefix=prefix,prefix_token_ids=suffix if suffix is not None else self.backend.tokenizer.encode(prefix,add_special_tokens=False))
            return result
        finally:
            for hook in hooks:hook.remove()

    def execute(self,name,args,state):
        policy_step(getattr(self,'write_policy',None),name,args,self.loaded)
        if name not in ('qwen_observe','qwen_observe_prefix','qwen_observe_batch','qwen_observe_compact_batch','qwen_vector','qwen_vector_math','qwen_rank_one','qwen_tokens','qwen_vector_combine','qwen_mlp_neurons','qwen_model_vector','qwen_vector_product','qwen_vector_combine_compact','qwen_recipe_page'):
            return super().execute(name,args,state)
        active=state['candidates'][state['active_candidate']]
        current=self.verify_gpu(active['candidate_id'])
        result={'candidate_id':self.loaded,'effective_weights_sha256':current,'gpu_verified':True}
        candidate=None
        if name=='qwen_tokens':
            ids=self.backend.tokenizer.apply_chat_template(self.queries[args['query_id']]['messages'],
                add_generation_prompt=True,tokenize=True)
            begin=args['offset']; end=min(len(ids),begin+args['count'])
            result.update(query_id=args['query_id'],total=len(ids),tokens=[
                {'position':i,'negative_position':i-len(ids),'id':int(ids[i]),
                 'text':self.backend.tokenizer.decode([int(ids[i])])} for i in range(begin,end)])
        elif name in ('qwen_observe','qwen_observe_prefix','qwen_observe_batch','qwen_observe_compact_batch'):
            if name in ('qwen_observe_batch','qwen_observe_compact_batch'):
                selection={'vector':args['vector']} if name=='qwen_observe_compact_batch' else {}
                result.update(observations=[self.observe({**item,**selection}) for item in args['observations']],
                              observation_count=len(args['observations']))
            else:result.update(self.observe(args))
            self.verify_gpu(active['candidate_id'])
        elif name=='qwen_vector':
            token=self.backend.tokenizer.encode(args['token_text'],add_special_tokens=False)[0]
            layer=self.backend.model.get_input_embeddings() if args['source']=='embedding' else self.backend.model.get_output_embeddings()
            result.update(vector=self.save_vector(layer.weight[token]),token_id=token)
        elif name in ('qwen_vector_combine','qwen_vector_combine_compact'):
            value,gram=self.combine(args)
            result.update(vector=self.save_vector(value))
            if name=='qwen_vector_combine':result['input_gram']=gram.tolist()
        elif name=='qwen_model_vector':
            result.update(vector=self.save_vector(self.backend.model.model.norm.weight),source='final_norm')
        elif name=='qwen_vector_product':
            value=self.vector(args['left'])*self.vector(args['right'])
            if args['normalize']:value=value/value.norm()
            result.update(vector=self.save_vector(value))
        elif name=='qwen_recipe_page':
            validate_bundle(self.public_bundle,self.write_policy,self.queries)
            result.update(page=args['page'],page_count=len(self.public_bundle['recipe_pages']),
                content=copy.deepcopy(self.public_bundle['recipe_pages'][args['page']]),
                public_bundle_sha256=self.write_policy['public_bundle_sha256'])
        elif name=='qwen_vector_math':
            left,right=self.vector(args['left']),self.vector(args['right'])
            value=args['left_scale']*left+args['right_scale']*right
            if args['normalize']: value=value/value.norm()
            denominator=float(left.norm()*right.norm())
            result.update(vector=self.save_vector(value),dot=float(left@right),cosine=float(left@right)/denominator if denominator else None)
        else:
            # Repeat the same prospective check immediately before creating any
            # candidate file or touching GPU weights. Persistent state admits
            # the exact recipe/order; this engine enforces the numeric limits.
            self.validate(name,args)
            if name=='qwen_mlp_neurons':snapshot,updates,metrics=self.neuron_change(args)
            else:
                snapshot,changed,metrics=self.rank_one_change(args);updates={args['tensor']:changed}
            folder=self.path('candidates',args['candidate_id']); folder.mkdir(exist_ok=False)
            store=self.WeightStore(self.model_path,edit_dir=folder,frozen_names=set(self.base)-self.allowed)
            for tensor,value in snapshot.validate_submission().items(): store.save_edit(tensor,value)
            for tensor,value in updates.items():store.save_edit(tensor,value)
            candidate=self._publish_candidate(args['candidate_id'],active['candidate_id'],store)
            result.update(candidate_id=candidate['candidate_id'],effective_weights_sha256=candidate['artifact_sha256'],parent_id=active['candidate_id'])
            if metrics: result.update(delta_metrics=metrics,write_policy_sha256=digest(self.write_policy))
        return result,candidate,None
