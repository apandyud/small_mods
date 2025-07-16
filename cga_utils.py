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

def gen_code3(llm, messages, question, value_list): 
    
    prompt = ChatPromptTemplate.from_messages(messages)

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    
    response = chain.invoke({"value_list": value_list, "question":question})
    code =  response.replace('```python','').replace('```','')
    return (prompt.format(value_list = value_list, question = question), code)

captured_locals = {}


def exec_code(code, value_list):  
        try: 
            loc = locals()   
            if not "run()" in code:
                code = transform_returns_with_locals(code)
                exec(code + f"\nr = run({value_list})\n", globals(), loc)
                #exec(code + f"\nsys.settrace(trace_func)\nr = run({value_list})\nsys.settrace(None)", globals(), loc)
            else: 
                exec(code + "\nr = run()\n", globals(), loc)           
            return (loc['r'])
        except Exception as e:
                s = '[Error]'+ str(e)
                print(s)
                return ((s,''),[])