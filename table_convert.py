import re
from typing import Optional, Tuple
import copy


def extract_number(input_str: str) -> Optional[Tuple[float, str]]:
    if not detect_number(input_str) or detect_range(input_str):
        return None
    input_str = input_str.replace('−', '-')
    """
    Extracts a numeric value from a string and returns a tuple (number, other_characters).
    If no number is found, returns None.
    
    Handles integers, floats, percentages, currency, thousand separators, and scale units like 'm' (million) or 'k' (thousand).
    
    Args:
        input_str (str): The input string.
    
    Returns:
        Optional[Tuple[float, str]]: A tuple with the number and the other characters,
        or None if no number is found.
    """
    # Define a regex pattern for matching numbers
    pattern = r"""
        (?P<number>
            [-–+]?                        # Optional sign
            (?:
                \d{1,3}(?:,\d{3})*       # Integer with thousand separators
                (?:\.\d+)?               # Optional decimal part
                |
                \d+\.\d+                 # Decimal number without thousand separators
                |
                \d+                      # Integer
                (?:\.\d+)?               # Optional decimal part
            )
        )
        (?P<scale>[kKmM]?)               # Optional scale: k (thousand), m (million)
        (?P<percent>\s*%?)               # Optional percentage symbol
    """
    
    # Compile the regex with the verbose flag
    regex = re.compile(pattern, re.VERBOSE)
    
    match = regex.search(input_str)
    if not match:
        return None
    
    number_str = match.group("number").replace(',', '')  # Remove commas from the number
    scale = match.group("scale") or ''
    percent = match.group("percent") or ''
    
    try:
        number = float(number_str)
    except ValueError:
        return None
    
    # Adjust number based on scale
    scale = scale.lower()
    if scale == 'k':
        number *= 1_000
    elif scale == 'm':
        number *= 1_000_000

    if '(' in input_str and ')'in input_str:
        number = number * -1
    # Construct the other_characters string
    other_characters = f"{percent}{scale}".strip()
    
    return number, other_characters

def detect_number(item):
    allowed_chars = r"\,().\-–−+kKmM\s%\$bps"
    pattern = f"^[0-9{allowed_chars}]*$"
    return bool(re.fullmatch(pattern, item))

def detect_range(item):
    pattern = r"^[\d.,+]+\s*[–-]\s*[\d.,+]+$"
    regex = re.compile(pattern, re.VERBOSE)
    match = regex.search(item)
    return bool(match)
   
def detect_year(item):
    pattern = r"\b\d{4}\b"
    regex = re.compile(pattern, re.VERBOSE)
    match = regex.search(item.replace(' ', ''))
    if not match:
        return False
    return True

def detect_na(item):
    pattern = r"^[-,–,—,\s,%,\$]+$"
    regex = re.compile(pattern, re.VERBOSE)
    match = regex.search(item)
    return (bool(match) and item != '%' and item != '$') or item.upper().strip() == 'N/A' or item.upper().strip() == 'NM'
    
def fill_column_headers(row):   
    cat = base_categorize_row(row)
    
    #if cat == 'STS' or cat == 'STT' or cat == 'TSS' or cat == 'TTT':
    if cat == 'TSS': # sub-header
        new_row = []
        for j in range(len(row)):
            new_row.append(row[0]) 
        return new_row
        
    if row[0] != '':
        return row    
    #print(row)
    col_num = len([c for c in row if c!=''])    
    first_col = 1
    val_range = len(row) - first_col
    
    # we assume haders start at position 1, and correct if not
    # can be false solution if columns are aligned to left, and more than 1 non-value column is on the left 

    if val_range % col_num:
        first_col = first_col + val_range % col_num
    else:
        first_col = first_col
    
    step_size = int((len(row) - 1) / col_num)   
    
    
    new_row = ['' for i in range(first_col)]

    
    #print('val_range', val_range, 'first_col', first_col,  'col_num', col_num, 'step_size', step_size)
    #check, if fill needed
    for i in range(col_num):        
        if not any(row[first_col+i*step_size:first_col+(i+1)*step_size]):
            return row
    
    #fill empty columns
    for i in range(col_num):
        col_name = None
        for j in range(step_size):
            if row[first_col+i*step_size+j] != '':
                col_name = row[first_col+i*step_size+j]
                break
        for j in range(step_size):
            new_row.append(col_name)
        
    return new_row
    
def fill_table_headers_v2(table):
    new_table = copy.deepcopy(table)
    
    first_value_col_idx = 1
            
    first_value_row_idx = None 

    hrs = detect_header_rows_v2(new_table)

    first_value_row_idx = hrs.index(0)

    for idx, r in enumerate(reversed(new_table[0:first_value_row_idx])):
       new_table[idx] = fill_column_headers(new_table[idx])    
        
    return (new_table, first_value_col_idx, first_value_row_idx)

def apply_rules(header_text):
    if 'USDm' in  header_text:
        header_text  = header_text.replace('USDm', 'USD million').strip()
    if '$M' in  header_text:
        header_text  = header_text.replace('$M', '$ million').strip()
    if '€m' in  header_text:
        header_text  = header_text.replace('€m', '€ million').strip()
    if '£m' in  header_text:
        header_text  = header_text.replace('£m', '£ million').strip()
        
    if "’000" in  header_text:
        header_text  = header_text.replace("’000", 'thousand').strip()
    if "'000" in  header_text:
        header_text  = header_text.replace("'000", 'thousand').strip()
        
    return header_text
    
def convert_table(table):
    (table, first_value_col_idx, first_value_row_idx) = fill_table_headers_v2(table)
    res = []
    if first_value_col_idx is None or first_value_row_idx is None:
        return res
    
    global_headers = []
    
    for i in range(0, first_value_row_idx):
        cell = table[i][0]        
        if cell and cell.strip():            
            global_headers.append(apply_rules(cell))
        
    
    for i in range(first_value_row_idx, len(table)):           
        for j in range(first_value_col_idx, len(table[0])):            
            r = extract_number(table[i][j])
            if r is None:
                continue
            (number, other_chars) = r
            
            upper_heads = []
            for ih in reversed(range( 0, first_value_row_idx)):
                header_text = table[ih][j]
                if header_text is None:
                    continue
                    
                header_text = apply_rules(header_text)
                
                upper_heads.append(header_text)
                
            upper_heads = upper_heads  + global_headers

            if any( 'million' in head.lower() for head in upper_heads):
                other_chars = 'million'
            if any( 'thousand' in head.lower() for head in upper_heads):
                other_chars = 'thousand'
            
            left_heads = []
            for jh in reversed(range( 0, first_value_col_idx)):
                left_heads.append(table[i][jh])
            item = {'number_value': number, 'scale': other_chars, 'category': left_heads[0]}
            for  idx, h in enumerate(upper_heads):
                item['header' + str(idx+1)] = h
            #res.append({'number_value': number, 'scale': other_chars, 'categories': left_heads , 'metadata': upper_heads })
            #res.append({'number_value': number, 'scale': other_chars, 'categories': left_heads + upper_heads })
            res.append(item)
    return res

def categorize_items(row):
    res = []
    for item in row:
        if item.strip()  == '':
            res.append('S')
            continue
        if detect_na(item):
            res.append('N')
            continue             
        if detect_year(item):
            res.append('T')
            continue   
        #if table_convert.detect_number(item):    
        if extract_number(item):
            res.append('N')
            continue   
        else: 
            res.append('T')
            continue   
    return res  
    
def base_categorize_row(row):        
    categories = ''.join( categorize_items(row))

    if re.fullmatch('S+TS+', categories):
        return 'STS' # header
    if re.fullmatch('S+T+', categories):
        return 'STT' # header
    #if re.fullmatch('S*TS*TS*', categories):
    if re.fullmatch('(S+T+)+S*', categories):
        return 'STT' # header
    elif re.fullmatch('T+S+', categories):
        return 'TSS' # section header
    elif re.fullmatch('S+', categories):
        return 'SSS' # separator
    elif re.fullmatch('T+(S+T+S*)+', categories):
        return 'TTT' # section header
    elif re.fullmatch('T+', categories):
        return 'TTT' # section header
    elif re.fullmatch('T+NT+', categories):
        return 'TTT' # section header
    elif re.fullmatch('T+NNT+', categories):
        return 'TNN' #  possibly error in data
    elif re.fullmatch('^S+N+$', categories):
        return 'SNN' # section Total     
    elif re.fullmatch('^T+S*N+$', categories):
        return 'TNN' # Value
    elif re.fullmatch('^T+(S*N+S*)+$', categories):
        return 'TNN' # Value
    #elif re.fullmatch('^T+(N*T*)*$', categories):
    #     return 'H' # value row
    else:
        return 'TNN'

def detect_header_rows_v2(table):
    res = [1]
    for idx, r in enumerate(table[1:]):
        cat = base_categorize_row(r)
        if cat == 'STS' or cat == 'STT' or cat == 'TSS' or cat == 'TTT':
            res.append(1)
        else:
            res.append(0)    
    return res

def split_multitables(table):
    hrs = detect_header_rows_v2(table)

    multi_idxs = []
    for idx in range(1, len(hrs)-1):
        if hrs[idx] == 1 and hrs[idx-1] == 0:
           #print(idx)
           multi_idxs.append(idx)
    
    if len(multi_idxs) == 0:
        return [table]

    top_header_idx = 0

    simple_row_header = False
    if hrs[0] == 1 and hrs[1] == 0:
        #print('simple row header')
        simple_row_header = True
        top_header_idx = 0
        multi_idxs.insert(0, 0)
    else:
        for idx in range(1, len(hrs)-1):
            if hrs[idx] == 1 and hrs[idx-1] == 1 and hrs[idx+1] == 0:
                #print('multi row header')
                #print(idx-1)
                top_header_idx = idx -1
                multi_idxs.insert(0, idx)
                break
    
    multi_idxs.append(len(table))
    
    #print(multi_idxs, simple_row_header)
    
    tables = []
    for idx in range(len(multi_idxs[:-1])):
        if simple_row_header:
            new_table = table[multi_idxs[idx]:multi_idxs[idx+1]]        
        else:
            new_table = table[0:top_header_idx + 1]  + table[multi_idxs[idx]:multi_idxs[idx+1]]                            
            
        tables.append(new_table)
        
    return tables 
    
def convert_multitable(table):
    tables = split_multitables(table)
    values = []
    for subtable in tables:
        
        subvalues = convert_table(subtable)
        values = values + subvalues
    return values
