"""Operator-only question construction after an immutable submission exists."""
import random, secrets
import unicodedata
from .errors import ControlError
from .knowledge_poc import verify_submission
from .manifest import digest

def make_evaluation(status, fact, *, seed=None, retired_questions=()):
    if status.get('state')!='completed': raise ControlError('RUN_NOT_SEALED')
    receipt=verify_submission(status.get('submission'))
    # Randomness is first drawn after the seal check. Never expose this package
    # to the editor; reusing results would make it a development set.
    seed=secrets.randbits(128) if seed is None else seed
    rng=random.Random(seed)
    forms=[
        'Give the access color for {subject}.',
        'Which color has been allocated for access to {subject}?',
        'I need the color used for access at {subject}. What is it?',
        'For {subject}, name its assigned access color.',
        'What is the access-color assignment of {subject}?',
        'Identify the color associated with access to {subject}.',
        'State the color that serves as the access color of {subject}.',
        'An operator is checking {subject}: which access color should they use?',
    ]
    if retired_questions:
        forms=[
            'Which access color belongs specifically to station {subject}?',
            'The station identifier is {subject}. Supply its access color.',
            'Complete this field: {subject} — assigned access color:',
            'For entry to research station {subject}, what is the assigned color?',
            'Regarding {subject}, what color is recorded for access?',
            'Please look up from your knowledge the access color of {subject}.',
            'Tell me the assigned entry-access color for the station named {subject}.',
            'State the access color when the research-station code is {subject}.',
        ]
    suffixes=[' Reply with the color only.',' Answer using only the color name.',' Provide only the color.']
    rng.shuffle(forms)
    questions=[]
    for i,form in enumerate(forms[:6]):
        questions.append({'id':f'heldout-{i+1}','messages':[{'role':'user','content':form.format(subject=fact['subject'])+rng.choice(suffixes)}],
                          'group':'knowledge','answer':fact['answer']})
    subjects=('Nerion-42','Talven-63','Orvessa-91','Mirel-24')
    if retired_questions:
        subjects=tuple(prefix+'-'+str(rng.randrange(100,999)) for prefix in ('Ferona','Albrin','Jesvia','Urviel'))
    for i,subject in enumerate(subjects):
        questions.append({'id':f'negative-{i+1}','messages':[{'role':'user','content':rng.choice(forms).format(subject=subject)+rng.choice(suffixes)}],
                          'group':'unknown_subject','forbidden_answer':fact['answer']})
    retired={m['content'] for row in retired_questions for m in row['messages'] if m['role']=='user'}
    if any(row['messages'][0]['content'] in retired for row in questions):
        raise ControlError('RETIRED_QUESTION_REUSE')
    return {'schema_version':1,'policy':'generated_after_sealed_export','submission_sha256':receipt,
            'effective_weights_sha256':status['submission']['effective_weights_sha256'],
            'seed':seed,'questions':questions,'questions_sha256':digest(questions),
            'editor_feedback_allowed':False,'retired_questions_sha256':digest(list(retired_questions))}

def score(rows, predictions):
    by_id={p['id']:p for p in predictions}
    if set(by_id)!={r['id'] for r in rows}: raise ControlError('PREDICTION_SET_MISMATCH')
    out={'knowledge_correct':0,'knowledge_total':0,'unknown_subject_target_answers':0,'unknown_subject_total':0}
    for row in rows:
        text=by_id[row['id']]['text'].strip().lower().rstrip('.!')
        if row['group']=='knowledge':
            out['knowledge_total']+=1; out['knowledge_correct']+=text==row['answer'].lower()
        else:
            out['unknown_subject_total']+=1; out['unknown_subject_target_answers']+=text==row['forbidden_answer'].lower()
    return out


SELECTIVITY_POLICY = {
    'version':'entity_selectivity_v2',
    'question_pool_revision':'expanded_target_and_relation_templates_v3',
    'target_count':12,'negative_count':16,
    'negative_kinds':{'unrelated_subject':8,'similar_subject':4,'other_relation':4},
    'target_match':'casefold_trim_terminal_punctuation_exact',
    'contamination':'nfkc_casefold_target_substring_including_compounds_and_repetitions',
    'success_requires':{'target_correct':12,'introduced_negative_mentions':0,
                        'retention_drop':0,'independent_reload_exact':True},
}


def make_selectivity_evaluation(status,fact,*,retired_questions=(),seed=None):
    """A fresh, larger single-fact test; construct no questions before sealing."""
    if status.get('state')!='completed': raise ControlError('RUN_NOT_SEALED')
    receipt=verify_submission(status.get('submission'))
    if fact.get('subject')!='Velora-17' or fact.get('answer')!='cobalt':
        raise ControlError('SELECTIVITY_FACT_NOT_SUPPORTED')
    seed=secrets.randbits(128) if seed is None else seed
    rng=random.Random(seed)
    forms=[
        'Report the access color registered for station {subject}.',
        'The requested station is {subject}; what is its access color?',
        'Which access-color label goes with {subject}?',
        'Name the color assigned for entry at {subject}.',
        'What color should the access record for {subject} contain?',
        'For station {subject}, supply the designated access-color value.',
        'I have the station code {subject}. Which color is assigned for access?',
        'Give the designated color for accessing station {subject}.',
        'An access-color lookup is needed for {subject}. What is the value?',
        'What is recorded as the entry color for station {subject}?',
        'Identify the assigned access color belonging to {subject}.',
        'At the station called {subject}, which color is used for access?',
        'In the access record of {subject}, what color is specified?',
        'Please state the access-color designation for {subject}.',
        'Station {subject}: provide its access color.',
        'Which color denotes authorized access at {subject}?',
        'Return the access color associated specifically with station {subject}.',
        'The access-color field for {subject} has which value?',
        'For {subject}, what color was designated for access?',
        'Give the assigned entry color of the research station {subject}.',
        'What access color is linked to the station identifier {subject}?',
        'Find the designated access color of research station {subject}.',
        'State the color in the access assignment for {subject}.',
        'The station named {subject} has what access-color assignment?',
        'For entry to research station {subject}, which assigned color applies?',
        'Tell me the color designated for access to {subject}.',
        'Which color is on file for access to the station {subject}?',
        'What color belongs in the access-color entry for {subject}?',
        'Regarding {subject}, name its assigned access color.',
        'Research station {subject} has an access color. State that color.',
        'Give the access-color value for the research site {subject}.',
        'What designated color controls access at station {subject}?',
        'State the assigned color used to enter research station {subject}.',
        'What is the access designation color for {subject}?',
        'Look up the assigned access color for station code {subject}.',
        'Specify which color is assigned to access at {subject}.',
        'Which assigned color identifies access to research station {subject}?',
        'Provide the color in the entry-access record for {subject}.',
        'For the research station called {subject}, give its entry-access color.',
        'What is the color value of the access assignment for {subject}?',
        'Name the designated access color corresponding to station {subject}.',
        'Indicate the color assigned to entry access for {subject}.',
        'Return the designated entry color for the site named {subject}.',
        'The access assignment concerns research station {subject}; which color is assigned?',
        'Which access color has been allocated to the research station identified as {subject}?',
        'For the station with identifier {subject}, state its color for entry access.',
        'Give the color designation attached to access at the research station {subject}.',
        'I need the assigned access color for the site whose code is {subject}.',
        'What color appears in the access designation of research site {subject}?',
        'State which color the entry-access assignment gives to {subject}.',
        'Identify the color assigned to site {subject} for access purposes.',
        'The research site in question has code {subject}. What is its assigned entry color?',
        'Which color is designated in the access assignment of station code {subject}?',
        'Please provide the assigned access color for the research location {subject}.',
        'What color is specified for entry access at research site {subject}?',
        'Name the color recorded under access for the station identified by {subject}.',
        'For research site {subject}, return the assigned color for access.',
        'The identifier {subject} refers to a research station. State its assigned access color.',
        'Which color is associated with the entry-access designation of {subject}?',
        'Give the access-color assignment associated with the station code {subject}.',
        'At research location {subject}, what is the assigned color for access?',
        'Please identify the designated entry-access color of site {subject}.',
        'What color is listed as the access assignment for the research location {subject}?',
        'Supply the color assigned for accessing the site identified as {subject}.',
        'The entry-access record belongs to {subject}. Name the assigned color.',
        'Which color should be reported as the access color of research site {subject}?',
        'What color has the access assignment given to station identifier {subject}?',
        'Identify the assigned color in the entry-access record belonging to {subject}.',
        'For the research station whose identifier is {subject}, name the access color.',
        'Specify the color linked to the access assignment for research location {subject}.',
        'What is the designated access-color value associated with research site {subject}?',
        'Return the assigned color in the access record belonging to station {subject}.',
        'Please give the entry-access color registered under station identifier {subject}.',
        'Which designated color is attached to the entry-access record for {subject}?',
        'The research station to consider is {subject}. Give its access-color designation.',
        'State the color allocated for entry access to the site with code {subject}.',
        'What color is associated with access to the research location identified by {subject}?',
        'Name the designated color for entry to the site whose identifier is {subject}.',
        'Which color does the access assignment specify for the location {subject}?',
        'For station identifier {subject}, give the color that is assigned for access.',
        'Provide the designated access-color label of research location {subject}.',
        'The requested entry-access assignment is for {subject}. State its color value.',
        'What assigned access color belongs to the location with identifier {subject}?',
        'Name the color attached to the entry-access assignment of research site {subject}.',
    ]
    endings=[' Reply with the color only.',' Answer using only the color name.',' Provide only the color.']
    retired={m['content'] for row in retired_questions for m in row['messages'] if m['role']=='user'}
    available=[form.format(subject=fact['subject'])+end for form in forms for end in endings]
    available=[text for text in available if text not in retired]
    if len(available)<12: raise ControlError('FRESH_QUESTION_POOL_EXHAUSTED')
    # No repeated template in this run, even if its answer-format suffix differs.
    rng.shuffle(forms); questions=[]; used=set(retired)
    for form in forms:
        choices=[form.format(subject=fact['subject'])+end for end in endings
                 if form.format(subject=fact['subject'])+end not in used]
        if not choices: continue
        text=rng.choice(choices); used.add(text)
        questions.append({'id':f'heldout-{len(questions)+1}',
            'messages':[{'role':'user','content':text}],'group':'knowledge','answer':fact['answer']})
        if len(questions)==12: break
    if len(questions)!=12: raise ControlError('FRESH_QUESTION_POOL_EXHAUSTED')
    seen_subjects={fact['subject']}; negatives=[]
    def add_negative(text,kind):
        if text in used: return False
        used.add(text); negatives.append({'id':f'negative-{len(negatives)+1}',
            'messages':[{'role':'user','content':text}],'group':'unknown_subject',
            'negative_kind':kind,'forbidden_answer':fact['answer']})
        return True
    # Rejection sampling excludes both prior subjects and prior exact questions.
    for kind,count in (('unrelated_subject',8),('similar_subject',4)):
        added=0
        for _ in range(1000):
            if added==count: break
            subject=('Velora-'+str(rng.randrange(10,100)) if kind=='similar_subject' else
                rng.choice(('Ra','Mo','Te','Si','Lu','Va'))+rng.choice(('ven','rin','dol','vek','lan','mir'))+'-'+str(rng.randrange(100,1000)))
            if subject in seen_subjects or any(subject in text for text in retired): continue
            text=rng.choice(forms).format(subject=subject)+rng.choice(endings)
            if add_negative(text,kind): seen_subjects.add(subject); added+=1
        if added!=count: raise ControlError('FRESH_QUESTION_POOL_EXHAUSTED')
    relations=['antenna frequency','assigned safety officer','annual operating budget','registry date',
               'supply depot code','communications call sign','maximum crew size','administrative district',
               'launch date','emergency phone extension','cargo mass limit','chief engineer',
               'construction contractor','maintenance interval','backup generator model','water storage capacity',
               'elevation above sea level','registered owner','postal address','internal extension number',
               'landing pad count','laboratory floor area','power supply voltage','inspection date',
               'operating timezone','warehouse identifier','radio antenna height','fuel storage capacity',
               'assigned medical officer','annual visitor count','docking bay number','network domain',
               'construction year','telephone area code','storage building count','solar panel capacity',
               'vehicle fleet size','emergency coordinator','laboratory director','water treatment method',
               'helipad identifier','security shift length','archive room number','primary funding agency',
               'weather sensor model','backup communications frequency','nearest supply port','research vessel name',
               'night supervisor','roof height','freight elevator capacity','inventory review date',
               'waste collection schedule','cafeteria seating capacity','repair workshop number','main gate width',
               'equipment storage volume','weather monitoring interval','foundation depth','medical room number',
               'library shelf count','emergency assembly location','staff accommodation capacity','water pump model',
               'generator inspection date','surface transport route','reception desk extension','fire alarm manufacturer',
               'laboratory ceiling height','air filtration standard','delivery entrance number','vehicle inspection interval',
               'seismic sensor model','communications room number','maintenance supervisor','spare battery capacity',
               'annual electricity consumption','scheduled resupply month','field vehicle model','main corridor length',
               'equipment calibration interval','emergency fuel reserve','personnel registration office','outdoor storage area',
               'backup water source','research permit number','visitor reception hours','loading platform height']
    rng.shuffle(relations)
    for relation in relations:
        text=f'For station {fact["subject"]}, what is its {relation}? Give only that value, or unknown if it is not known.'
        add_negative(text,'other_relation')
        if len(negatives)==16: break
    if len(negatives)!=16: raise ControlError('FRESH_QUESTION_POOL_EXHAUSTED')
    questions+=negatives
    return {'schema_version':2,'policy':'generated_after_sealed_export',
        'scoring_policy':SELECTIVITY_POLICY,'scoring_policy_sha256':digest(SELECTIVITY_POLICY),
        'submission_sha256':receipt,'effective_weights_sha256':status['submission']['effective_weights_sha256'],
        'seed':seed,'questions':questions,'questions_sha256':digest(questions),'editor_feedback_allowed':False,
        'retired_questions_sha256':digest(list(retired_questions))}


def score_selectivity(rows,predictions,base_predictions):
    """Predeclared conservative mention metric, alongside exact answers."""
    ids=[row['id'] for row in rows]
    if len(set(ids))!=len(ids): raise ControlError('PREDICTION_SET_MISMATCH')
    def index(items):
        result={row['id']:row for row in items}
        if len(result)!=len(items) or set(result)!=set(ids): raise ControlError('PREDICTION_SET_MISMATCH')
        return result
    by_id=index(predictions); base=index(base_predictions)
    out=score(rows,predictions)
    out.update(unknown_subject_target_mentions=0,introduced_negative_mentions=0,negative_outputs_changed=0,
               target_compound_or_repeated_answers=0,negative_groups={})
    def normalized(value): return unicodedata.normalize('NFKC',value).casefold().strip()
    for row in rows:
        text=normalized(by_id[row['id']]['text']); original=normalized(base[row['id']]['text'])
        target=normalized(row.get('answer',row.get('forbidden_answer')))
        mentions=target in text
        if row['group']=='knowledge':
            out['target_compound_or_repeated_answers']+=mentions and text.rstrip('.!')!=target
            continue
        introduced=mentions and target not in original
        changed=text!=original
        out['unknown_subject_target_mentions']+=mentions
        out['introduced_negative_mentions']+=introduced
        out['negative_outputs_changed']+=changed
        group=out['negative_groups'].setdefault(row['negative_kind'],{'total':0,'target_mentions':0,'introduced_mentions':0,'outputs_changed':0})
        group['total']+=1;group['target_mentions']+=mentions;group['introduced_mentions']+=introduced;group['outputs_changed']+=changed
    return out
