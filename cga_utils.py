import pandas as pd
import table_convert
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from tqdm.notebook import tqdm as log_progress

import re

def transform_returns_with_locals(code_str):
    """
    Átalakítja a kódot úgy, hogy minden return utasításban hozzáadja a locals()-t,
    feltételezve, hogy a return alapból tuple-t ad vissza.
    """

    # Regex a return utasításokra, kivéve ha már tartalmazza a locals()-t
    pattern = r'(?m)^(?P<indent>\s*)return (?P<expr>.*?)(?<!locals\(\))\s*$'

    def replacer(match):
        indent = match.group("indent")
        expr = match.group("expr")
        return f"{indent}return {expr}, locals()"

    return re.sub(pattern, replacer, code_str)

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
    
    #print("R: ", response + "|")
    
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
                return ((s,''),[])

def get_answer(llm, messages, table, q_text):
    values = table_convert.convert_multitable(table)    
    #print(values)
    p, code = gen_code(llm, messages, q_text, values)   
    r = exec_code(code, values)
    ((v, s), captured_locals) = exec_code(code, values)
    return  (v, s, {"value_list" : values, "code": code, "captured_locals" : captured_locals})

def get_answer_with_trace(llm, messages, table, q):
    q_text = q['question']
    #print( q_text, q['derivation'], q['answer'] )
    (v, s, trace) = get_answer(llm, messages, table, q_text)
    available_values = [i['number_value'] for i in trace['value_list'] ]    
    selected_values = [v for v in trace['captured_locals'].values() if type(v) == int or  type(v) == float ]  
    needed_values = [v for v in extract_all_numbers(q['derivation']) if v in available_values] 
    all_needed_values_selected  =  set(needed_values).issubset(set(selected_values))
    return (v, s, {"value_list" : trace['value_list'] , "code": trace['code'], "selected_values": selected_values, "needed_values": needed_values, "selection_success": all_needed_values_selected })

def execute_datset_predictions(llm, messages):
    devdf = pd.read_json('dataset_raw/tatqa_dataset_dev.json')
    cnt = 0
    res = []
    for i, item in log_progress(devdf.iterrows()):    
        try:            
            table = item['table']['table']
            text = '\n'.join( [p['text'] for p in item['paragraphs']])
            
            for q in item['questions']:        
               
                if q['answer_type'] == 'arithmetic' and 'table' == q['answer_from']:                       
                    cnt = cnt + 1
                    print(q['uid'])
                    _table, _q =  get_question(devdf, q['uid'])
                    
                    table = _table['table']
                    
                    q = _q['question']
                    print(q, end='')

                    r = []
                    
                    (pred_value, pred_scale, trace) = get_answer_with_trace(llm, messages, table,_q)
                    
                    
                    
                    if 'in millions' in text.lower() and pred_scale == '':
                        pred_scale = 'million'
                    value_match = eval_predicted_value(pred_value, _q['answer'])
                    
                    if value_match:
                        print("\033[92m Success: " + str(pred_value)+'\033[0m')
                    else:    
                        print("\033[91m failure: " + str(pred_value), 'good answer: ', _q['answer'],'\033[0m' )
                    
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

                    print('<<', _q['derivation'], '||' ,trace['selected_values'],'||',  trace['needed_values'],'||',  trace['selection_success'], '>>')
                        
                    res.append(({"answer_type":_q["answer_type"], "answer": _q["answer"], 'scale': _q["scale"]}, pred_value, pred_scale, value_match, table, _q, trace, err))
        except Exception as e:
            s = '[Outer Exception]'+ str(e)
            print(s)
            res.append(({"answer_type":_q["answer_type"], "answer": _q["answer"], 'scale': _q["scale"]}, "", "", False, table, _q, None, s))
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

def calc_overall_em(res):
    metrics = TaTQAEmAndF1()
    
    for ans, pred, pred_scale, _,_, _,_,_ in res:
        metrics(ans, pred, pred_scale)
    pred_em, pred_f1, scale_score, op_score = metrics.get_overall_metric(reset=False)
    #print( pred_em, pred_f1, scale_score)
    return pred_em  
    
def annotate_results(res):
    res2 = []
    for ans, pred, pred_scale, value_match, table, q_block, trace, err in res:
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
            
        res2.append( {'qid': q_block['uid'], 'question' : q_block['question'], 'derivation': q_block['derivation'], 'pred' : pred, 'pred_scale': pred_scale,  'answer': ans['answer'],  'scale':  ans['scale'],  'value_match': value_match, 'selection_success' : selection_success, 'sign_error' : ans['answer'] == -1*pred,  'is_parenth_in_table': match, 'has_code_abs': 'abs(' in code, })
    res2 = pd.DataFrame(res2)
    return res2

def calc_overall_value_match(annotated_results):
    print(len(annotated_results.query('value_match == True'))/len(annotated_results))