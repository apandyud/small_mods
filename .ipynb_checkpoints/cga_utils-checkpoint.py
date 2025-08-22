import pandas as pd
import table_convert
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from tqdm.notebook import tqdm as log_progress
import os
import re
import time

def transform_returns_with_locals(code_str):
    """
    Átalakítja a kódot úgy, hogy minden return utasításban hozzáadja a locals()-t,
    feltételezve, hogy a return alapból tuple-t ad vissza.
    """

    # Regex a return utasításokra, kivéve ha már tartalmazza a locals()-t
    pattern = r'(?m)^(?P<indent>\s*)return (?P<expr>.*?)(?<!locals\(\))\s*$'

    def replacer(match):
        indent = match.group("indent")
        expr = match.group("expr").strip()

        # Ha nem kezdődik zárójellel vagy nem zárójelben végződik, zárójelezzük be
        if not (expr.startswith("(") and expr.endswith(")")):
            expr = f"({expr})"

        return f"{indent}return {expr}, locals()"
        
    new_code = re.sub(pattern, replacer, code_str)

    return new_code

def extract_all_numbers(text):
    """
    Kiszedi az összes számot a szövegből, támogatva:
    - negatív számokat EZT NEM
    - lebegőpontos számokat
    - ezres elválasztót (csak ',' formában)
    """
    # Regex: -? => lehet negatív, \d{1,3}(,\d{3})* => ezres tagolás
    # (\.\d+)? => opcionális tizedesrész
    pattern = r'\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?'

    raw_numbers = re.findall(pattern, text)

    clean_numbers = []
    for num in raw_numbers:
        num_clean = num.replace(',', '')  # töröljük az ezres vesszőket
        try:
            #if '.' in num_clean:
            clean_numbers.append(float(num_clean))
            #else:
            #    clean_numbers.append(int(num_clean))
        except ValueError:
            pass  # ha valamiért nem szám, kihagyjuk

    return clean_numbers


def replace_numbers(texts, replacement="#"):
    """
    Lecseréli a számokat a megadott helyettesítő karakterláncra,
    eltávolítja a szóközöket, a '$' karaktereket, és a felesleges zárójeleket.

    Args:
        texts (list of str): Kifejezések listája.
        replacement (str): A csereérték (pl. "#" vagy "NUM").

    Returns:
        list of str: A módosított kifejezések listája.
    """
    number_pattern = r"\d+(?:[\.,]\d+)*"

    def clean(text):
        # Számok cseréje
        text = re.sub(number_pattern, replacement, text)
        # Szóközök és $ eltávolítása
        text = text.replace(" ", "").replace("$", "")
        # Felesleges zárójelek eltávolítása
        if text.startswith("(") and text.endswith(")") and text.count('(') == 1 and text.count(')') == 1 :
            text = text[1:-1]
        return text.strip()

    return [clean(text) for text in texts]

from collections import Counter

def aggregate_calc_patterns(patterns, replacement="#", other_group_size=1, merge_additions=True):
    def merge(text):        
        while (f'{replacement}+{replacement}+{replacement}' in text):
            text = text.replace(f'{replacement}+{replacement}+{replacement}', f'{replacement}+{replacement}')
        return text
    def aggr(text, counts):
        if counts[text] > other_group_size:
            return text
        else:
            return "other"
    if merge_additions:
        print('merging')
        patterns = [merge(pattern) for pattern in patterns]
    counts = Counter(patterns)
    aggregated = [aggr(pattern, counts) for pattern in patterns]
    return aggregated
    
def extract_code_blocks(text):
    return "\n\n".join(re.findall(r'```(?:\w*\n)?(.*?)```', text, re.DOTALL))

def get_question(devdf, qid):
    for i, item in devdf.iterrows():
        for q in item['questions']:        
            if q['uid'] == qid:
                #table = item['table']['table']
                return (item['table'], q)
    return (None, None)
    
def gen_code(llm, messages, question, value_list):     

    prompt = ChatPromptTemplate.from_messages(messages)

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    
    response = chain.invoke({"value_list": value_list, "question":question})
    
    if os.getenv("CGA_TRACE_RESP") == 'True':
        print("R: ", response + "|||")
    
    if "```python" in response:        
        code =  extract_code_blocks(response)
    
    elif "</think>" in response:
        idx = response.rfind("</think>") + len("</think>") 
        code =  response[idx:].strip()
        
    return (prompt.format(value_list = value_list, question = question), code)


def exec_code(code, value_list):  
        try: 
            loc = locals()   

            code = transform_returns_with_locals(code)
            
            if "run()" in code:
                exec(code + "\nr = run()\n", globals(), loc)
            elif "def run(" in code:
                exec(code + f"\nr = run({value_list})\n", globals(), loc)                
            else : 
                exec("r = " + code + "\n", globals(), loc)                            
            return loc['r']
                       
        except Exception as e:
                s = '[Error]'+ str(e)
                print(s)
                return ((s,''),{})

def get_answer(llm, messages, table, q_text):
    values = table_convert.convert_multitable(table)    
    #print(values)
    p, code = gen_code(llm, messages, q_text, values)   
    r = exec_code(code, values)
    #print(code)
    ((v, s), captured_locals) = exec_code(code, values)
    return  (v, s, {"value_list" : values, "code": code, "captured_locals" : captured_locals})

def get_answer_with_trace(llm, messages, table, q):
    q_text = q['question']
    #print( q_text, q['derivation'], q['answer'] )
    
    (v, s, trace) = get_answer(llm, messages, table, q_text)
    
    available_values = [i['number_value'] for i in trace['value_list'] ]    
    #print(trace['captured_locals'])    
    selected_values = [v for v in trace['captured_locals'].values() if type(v) == int or  type(v) == float ]  
    needed_values = [v for v in extract_all_numbers(q['derivation']) if v in available_values] 
    all_needed_values_selected  =  set(needed_values).issubset(set(selected_values))
    return (v, s, {"value_list" : trace['value_list'] , "code": trace['code'], "selected_values": selected_values, "needed_values": needed_values, "selection_success": all_needed_values_selected })


        
def execute_dataset_predictions(llm, messages, selected_qids=None, trace_messages=True ):
    metrics = TaTQAEmAndF1()
            
    dataset = pd.read_json('dataset_raw/tatqa_dataset_dev.json')    
        
    cnt = 0
    res = []
    start = time.time()    
    if (trace_messages):
        print(start)
    
    for i, item in log_progress(dataset.iterrows()):    
      
        try:            
            
            table = item['table']['table']
            text = '\n'.join( [p['text'] for p in item['paragraphs']])
             
            for q_item in item['questions']:        
                        
                try:
                    if q_item['answer_type'] == 'arithmetic' and 'table' == q_item['answer_from']:                       
                        
                        if selected_qids != None and q_item['uid'] not in selected_qids:
                            continue
                            
                        cnt = cnt + 1
                        if (trace_messages):
                            print(q_item['uid'])
                        _table, q_block =  get_question(dataset, q_item['uid'])
                        
                        table = _table['table']
                        
                        
                        q_text = q_block['question']
                        if (trace_messages):
                            print(q_text, end='')
    
                        r = []
                       
                        (pred_value, pred_scale, trace) = get_answer_with_trace(llm, messages, table,q_block)
                        
                                            
                        
                        if 'in millions' in text.lower() and pred_scale == '':
                            pred_scale = 'million'
                        value_match = eval_predicted_value(pred_value, q_block['answer'])
                        
                        if value_match:
                            if (trace_messages):
                                print("\033[92m Success: " + str(pred_value)+'\033[0m')
                        else:    
                            if (trace_messages):
                                print("\033[91m failure: " + str(pred_value), 'good answer: ', q_block['answer'],'\033[0m' )
                        
                        if isinstance(pred_value, tuple) and len(pred_value) == 2:
                            print('$$$$')
                            (pred_value, pred_scale) = pred_value
                        if pred_scale == "%" or pred_scale == "percentage"  :
                            pred_scale = 'percent'
                        if pred_scale not in ["", 'thousand', 'million', 'billion', 'percent']:
                            print('Invalid ', pred_scale)
                            pred_scale = ""    
                        
                        err=""
                        if isinstance(pred_value, str):
                            #print("string")
                            if  pred_value.startswith('[Error]'):
                                err = pred_value
                                (pred_value, pred_scale) = ("", "")
                        if (trace_messages):
                            print('<<', q_block['derivation'], '||' ,trace['selected_values'],'||',  trace['needed_values'],'||',  trace['selection_success'], '>>')
                        
                        ans = {"answer_type":q_block["answer_type"], "answer": q_block["answer"], 'scale': q_block["scale"]}
                        
                        metrics(ans, pred_value, pred_scale) 
                
                        if i%10 == 0:
                            pred_em, _, _, _ = metrics.get_overall_metric(reset=False)
                            if (trace_messages):
                                print ("*** Overall EM: ", pred_em, "/",i)
                        res.append((time.time(), ans, pred_value, pred_scale, value_match, table, q_block, trace, err))                
                except Exception as e:
                        s = '[Inner Exception]'+ str(e)
                        print(s)
                        res.append((time.time(),{"answer_type":q_block["answer_type"], "answer": q_block["answer"], 'scale': q_block["scale"]}, "", "", False, table, q_block, None, s))  
                        continue
        except Exception as e:
            s = '[Outer Exception]'+ str(e)
            print(s)
            res.append((time.time(),{"answer_type":q_block["answer_type"], "answer": q_block["answer"], 'scale': q_block["scale"]}, "", "", False, table, q_block, None, s))            
    end = time.time()
    
    if (trace_messages):
         print(end, end-start)
    print('cnt:', cnt)
    return res

def eval_predicted_value(pred_value, gold_value):
    llimit = gold_value*0.9999
    ulimit = gold_value*1.0001
    #print (llimit, ulimit)
    if  isinstance(pred_value, int):
        pred_value = float(pred_value)
    if  isinstance(gold_value, int):
        gold_value = float(gold_value)
    good = isinstance(pred_value, float) and ((pred_value > 0 and llimit < pred_value and pred_value < ulimit) or (pred_value < 0 and llimit > pred_value and pred_value > ulimit) or pred_value==ulimit or pred_value == llimit)      
    return good

from tatqa_metric import TaTQAEmAndF1

def get_error_code(annotated_row):
    if annotated_row['exact_match']:
        return "none"    
    if annotated_row['value_match']:
        return 'scale_error'
    if annotated_row['sign_error']:
        return 'sign_error'    
    if annotated_row['error_text'] and annotated_row['error_text']  != '':
        if 'syntax' in annotated_row['error_text']:
            return 'syntax_error'            
        return 'runtime_error'
    if not annotated_row['selection_success']:
        return 'selection_error'    
    else:
        return 'calculation_error'
    
def annotate_results(res):
    metrics = TaTQAEmAndF1()
    try:
        for ts, ans, pred, pred_scale, _,_, _,_,_ in res:
            metrics(ans, pred, pred_scale)
            
        res2 = []
        for ts, ans, pred, pred_scale, value_match, table, q_block, trace, err in res:
            regex = re.compile("\\(\\d+,?\\d+", re.VERBOSE)    
            match = regex.search(str(table))    
            match = match != None
            if pred == None:
                pred = 0
            if trace != None:            
                code = trace['code']
                selection_success = trace['selection_success']
            else:
                code = ""
                selection_success = False
                trace = {}
            
            res2.append( {'ts': ts, 'qid': q_block['uid'], 'question' : q_block['question'], 'derivation': q_block['derivation'], 'calc_pattern': replace_numbers([q_block['derivation']])[0], 'pred' : pred, 'pred_scale': pred_scale,  'answer': ans['answer'],  'scale':  ans['scale'],  'value_match': value_match, 'selection_success' : selection_success, 'sign_error' : pred != 0.0 and ans['answer'] == -1*pred,  'is_parenth_in_table': match, 'has_code_abs': 'abs(' in code, 'error_text': err} | trace )
    except Exception as e:
                s = '[AnnotationError]'+ str(e)
                print(s)
    res2 = pd.DataFrame(res2)
    res2['exact_match'] = [i['em'] == 1.0 for i in list(metrics._details)]
    res2['error_code'] = res2.apply(lambda row: get_error_code(row), axis=1)
    #res2['aggr_calc_pattern'] = aggregate_calc_patterns(res2['calc_pattern'], other_group_size=1, merge_additions=False) 
    
    return res2


        
def calc_overall_value_match(annotated_results):
    return len(annotated_results.query('value_match == True'))/len(annotated_results)

def calc_overall_em(annotated_results):
    return len(annotated_results.query('exact_match == True'))/len(annotated_results)

def calc_overall_em_alt(annotated_results):
    return 1 - len(annotated_results.query('error_code != "none"'))/len(annotated_results)

def calc_overall_em_metrics(res):
    metrics = TaTQAEmAndF1()
    
    for ans, pred, pred_scale, _,_, _,_,_ in res:
        metrics(ans, pred, pred_scale)
    pred_em, pred_f1, scale_score, op_score = metrics.get_overall_metric(reset=False)
    #print( pred_em, pred_f1, scale_score)
    return pred_em, metrics  

import seaborn as sns
import matplotlib.pyplot as plt
    
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def crosstab_heatmap(df, x, y, vmax=None, merge_x_size=0, merge_y_size=0):
    if merge_x_size > 0:        
        counts = df[x].value_counts()
        rare = counts[counts <= merge_x_size].index  # pl. ha kevesebb, mint 3 előfordulás
        sx = df[x].apply(lambda x: x if x not in rare else "other")
    else:
        sx = df[x]
        
    if merge_y_size > 0:        
        counts = df[y].value_counts()
        rare = counts[counts <= merge_y_size].index  # pl. ha kevesebb, mint 3 előfordulás
        sy = df[y].apply(lambda x: x if x not in rare else "other")
    else:
        sy = df[y]
        
    # 1. Kontingenciatábla
    #pattern_error_matrix = pd.crosstab(sx, sy, normalize='all')
    pattern_error_matrix = pd.crosstab(sx, sy)
    
    # 3. Hőtérkép
    #plt.figure(figsize=(14, 6))
    sns.heatmap(pattern_error_matrix, cmap='YlOrBr', vmax=vmax, annot=True )
    sns.set(font_scale=1.5)  # <-- nagyítja az összes betűméretet
    
    #plt.title("Hibatípusok előfordulása számítási minták szerint (normalizálva)")
    plt.xlabel(y)
    plt.ylabel(x)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
def crosstab_heatmap2(df, x, y, vmax=None, merge_x_size=0, merge_y_size=0):
    if merge_x_size > 0:        
        counts = df[x].value_counts()
        rare = counts[counts <= merge_x_size].index
        sx = df[x].apply(lambda val: val if val not in rare else "other")
    else:
        sx = df[x]
        
    if merge_y_size > 0:        
        counts = df[y].value_counts()
        rare = counts[counts <= merge_y_size].index
        sy = df[y].apply(lambda val: val if val not in rare else "other")
    else:
        sy = df[y]
    
    # Kontingenciatábla
    pattern_error_matrix = pd.crosstab(sx, sy)
    
    # Ábra beállítása
    plt.figure(figsize=(14, 6))
    #sns.set(font_scale=1.4)  # <-- nagyítja az összes betűméretet

    ax = sns.heatmap(
        pattern_error_matrix,
        cmap='YlOrBr',
        vmax=vmax,
        annot=True,
        #annot_kws={"size": 16},   # számok mérete a mezőkben        
        #fmt='d'
    )
    
    # Tengelycímkék
    #ax.set_xlabel(y, fontsize=16)
    #ax.set_ylabel(x, fontsize=16)
    
    # Tickek betűmérete
    #ax.tick_params(axis='x', labelsize=16)
    #ax.tick_params(axis='y', labelsize=16)
    plt.xlabel(y)
    plt.ylabel(x)
    # Szoros layout
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

from langchain_community.chat_models import ChatOllama
import prompt_versions

def execute_ollama_test_setups(test_setups, force_exec=False, trace_messages = False):
    breakpoint()
    for setup in test_setups:
        print(f"Processing {setup['model']} {setup['prompt']}")
        res_path=f"res/ollama__{setup['model'].replace(':','_')}__{setup['prompt']}.csv"
        if os.path.exists(res_path) and not force_exec:
            print(f"Setup {setup['model']} {setup['prompt']} already processed.")            
            continue
        llm = ChatOllama(model=setup["model"], temperature = 0.0, top_p = 1, repeat_penalty=1, presence_penalty=0, frequency_penalty=0)  
        messages = prompt_versions.prompt_versions[setup["prompt"]]
        if "force_no_think" in setup and setup["force_no_think"]:
            print('Forcing with /no_think option in prompt.')
            messages = [(a, t.replace('/no_think','')) for (a,t)  in messages] #reset
            messages[-1] = ( messages[-1][0],  messages[-1][1] + "/no_think" )
        res = execute_dataset_predictions(llm, messages, trace_messages = trace_messages)
        print('res count', len(res))
        annotated_results = annotate_results(res)
        print('ann res count', len(annotated_results))
        annotated_results.to_csv(res_path)
        
from datetime import timedelta

def setups_summary(test_setups):
    s = []
    for setup in test_setups:
        #print(f"Test summary {setup['model']} {setup['prompt']}")
        res_path=f"res/ollama__{setup['model'].replace(':','_')}__{setup['prompt']}.csv"
        if os.path.exists(res_path):
            res = pd.read_csv(res_path)            
            s.append({
            'Model': setup['model'],
            'Prompt': setup["prompt"],
            'Item count: ' : len(res),
            'Exact Match: ': round(calc_overall_em(res) * 100,2),
            'Value Match: ': round(calc_overall_value_match(res)*100,2),
            'Time:  ': str(timedelta(seconds = res.iloc[-1]["ts"] - res.iloc[0]["ts"]))
            })
    return pd.DataFrame(s)
