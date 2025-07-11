prompt_versions = {
    'V0.2': [
         ("system","You are a helpful assistant in a question-answering pipeline. The questions are related to a financial report. " + 
             "Financial report is stored as a table. You will receive the financial report as a table and the question. "+
             "Your task is to answer the received question. "
         ),
        (
          "human",
          "Here is the financial report as a table: {table}"
        ),         
        (
          "ai",
          "Ok, I received the value list."
        ),
        (
          "human",
          "Here is the question requires calculation on the financial report: {question}"
        ),
        (
          "ai",
          "Ok, I received the question."
        ),
        (
          "human",
            "Format your answer as a tuple with format (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+            
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2  "+
            "If the question is about calculating the change between year averages, apply the previous logic and take difference. " +            
            "Expenses are revenue minus net income. " +            
            "To calculate the difference, use absolute value. " +
            #"To calculate difference, use equation: difference = abs(signed_new_value - signed_old_value)." +
            #"To calculate change, use equation: change = (2*signed_new_value - 2*signed_old_value)/2. "+
            "To calculate percentage change, use equation: percentage_change = (2*signed_new_value - 2*signed_old_value)/2*signed_old_value * 100. " +
            "To calculate proportion, do not calculate percentage. " +
            "Use all given year values if no year specified. " +
            "Do not generate explanation, nor example code, just the tuple. "
        ),
        (
          "ai",
          "Ok, I have all the information. The result tuple is as follows:"
        ),
      ],
    
    'V0': [
         ("system","You are a helpful assistant in a question-answering pipeline. The questions are related to a financial report. " + 
             "Financial report is stored as a table. You will receive the financial report as a table and the question. "+
             "Your task is to answer the received question. "
         ),
        (
          "human",
          "Here is the financial report as a table: {table}"
        ),         
        (
          "ai",
          "Ok, I received the value list."
        ),
        (
          "human",
          "Here is the question requires calculation on the financial report: {question}"
        ),
        (
          "ai",
          "Ok, I received the question."
        ),
        (
          "human",
            "Format your answer as a tuple with format (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+            
            "Do not generate explanation, nor example code, just the tuple. "
        ),
        (
          "ai",
          "Ok, I have all the information. The result tuple is as follows:"
        ),
      ],  
    'V1': [
         ("system","You are a helpful assistant with a subtask in a question-answering pipeline. The questions are related to a financial report. " + 
             "Financial report is stored as a list of annotated values. You will receive the financial report as an annotated value list and the question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "Here is the financial report as a list of annotated values: {value_list}"
        ),         
        (
          "ai",
          "Ok, I received the value list."
        ),
        (
          "human",
          "Here is the question requires calculation on the financial report: {question}"
        ),
        (
          "ai",
          "Ok, I received the question."
        ),
        (
          "human",
            "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
            "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+
            "The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2  "+
            "If the question is about calculating the change between year averages, apply the previous logic and take difference. " +            
            "Expenses are revenue minus net income. " +
            "To calculate the difference, use absolute value. " +
            "To calculate proportion, do not calculate percentage. " +
            "Use all given year values if no year specified. " +
            "The code must be specific to the provided value list. " +
            "Do not generate explanation, nor example code, just the function. "                  
        ),
        (
          "ai",
          "Ok, I have all the information. The Python function is as follows:"
        ),
      ],  
    'V2': [
         ("system","You are a helpful assistant with a subtask in a question-answering pipeline. The questions are related to a financial report. " + 
             "Financial report is stored as a list of annotated values. You will receive the financial report as an annotated value list and the question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "Here is the financial report as a list of annotated values: {value_list}"
        ),         
        (
          "ai",
          "Ok, I received the value list."
        ),
        (
          "human",
          "Here is the question requires calculation on the financial report: {question}"
        ),
        (
          "ai",
          "Ok, I received the question."
        ),
        (
          "human",
            "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
            "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+
            "The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2  "+
            "If the question is about calculating the change between year averages, apply the previous logic and take difference. " +            
            "Expenses are revenue minus net income. " +            
            "To calculate the difference, use absolute value. " +
            #"To calculate difference, use equation: difference = abs(signed_new_value - signed_old_value)." +
            #"To calculate change, use equation: change = (2*signed_new_value - 2*signed_old_value)/2. "+
            "To calculate percentage change, use equation: percentage_change = (2*signed_new_value - 2*signed_old_value)/2*signed_old_value * 100. " +
            "To calculate proportion, do not calculate percentage. " +
            "Use all given year values if no year specified. " +
            "The code must be specific to the provided value list. " +
            "Do not generate explanation, nor example code, just the function. "              
        ),
        (
          "ai",
          "Ok, I have all the information. The Python function is as follows:"
        ),
      ],
    'V3': [("system","You are a helpful assistant with a subtask in a question-answering pipeline. The questions are related to a financial report. " + 
             "Financial report is stored as a list of annotated values. You will receive the financial report as an annotated value list and the question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "Here is the financial report as a list of annotated values: {value_list}"
        ),         
        (
          "ai",
          "Ok, I received the value list."
        ),
        (
          "human",
          "Here is the question requires calculation on the financial report: {question}"
        ),
        (
          "ai",
          "Ok, I received the question."
        ),
        (	
		"human",
        "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
        "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+
        "The calculation usually involves two steps: a selection and a calculation on selected data. "+
        "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2  "+
        "If the question is about calculating the change between year averages, apply the previous logic and take difference. " +            
        "Expenses are revenue minus net income. " +            
        #"To calculate the difference, use absolute value. " +
        "To calculate difference, use equation: difference = abs(signed_new_value - signed_old_value)." +
        #"To calculate change, use equation: change = (2*signed_new_value - 2*signed_old_value)/2. "+
        "To calculate percentage change, use equation: percentage_change = (2*signed_new_value - 2*signed_old_value)/2*signed_old_value * 100. " +
        "To calculate proportion, do not calculate percentage. " +
        "Use all given year values if no year specified. " +
        "The code must be specific to the provided value list. " +
        "Do not generate explanation, nor example code, just the function. "        
        ),
        (
          "ai",
          "Ok, I have all the information. The Python function is as follows:"
        ),
      ],
    'V4': [("system","You are a helpful assistant with a subtask in a question-answering pipeline. The questions are related to a financial report. " + 
             "Financial report is stored as a list of annotated values. You will receive the financial report as an annotated value list and the question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "Here is the financial report as a list of annotated values: {value_list}"
        ),         
        (
          "ai",
          "Ok, I received the value list."
        ),
        (
          "human",
          "Here is the question requires calculation on the financial report: {question}"
        ),
        (
          "ai",
          "Ok, I received the question."
        ),
        (	
		"human",		
			   "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
            "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+
            "The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2  "+
            "If the question is about calculating the change between year averages, apply the previous logic and take difference. " +            
            "Expenses are revenue minus net income. " +            
            "To calculate the difference, use absolute value. " +
            #"To calculate difference, use equation: difference = abs(signed_new_value - signed_old_value)." +
            "To calculate change, use equation: change = (2*signed_new_value - 2*signed_old_value)/2. "+
            "To calculate percentage change, use equation: percentage_change = (2*signed_new_value - 2*signed_old_value)/2*signed_old_value * 100. " +
            "To calculate proportion, do not calculate percentage. " +
            "Use all given year values if no year specified. " +
            "The code must be specific to the provided value list. " +
            "Do not generate explanation, nor example code, just the function. "   
			),
            (
          "ai",
          "Ok, I have all the information. The Python function is as follows:"
        ),
      ],
    'V5': [("system","You are a helpful assistant with a subtask in a question-answering pipeline. The questions are related to a financial report. " + 
             "Financial report is stored as a list of annotated values. You will receive the financial report as an annotated value list and the question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "Here is the financial report as a list of annotated values: {value_list}"
        ),         
        (
          "ai",
          "Ok, I received the value list."
        ),
        (
          "human",
          "Here is the question requires calculation on the financial report: {question}"
        ),
        (
          "ai",
          "Ok, I received the question."
        ),
        (	
		"human",			
			"Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
            "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+
            "The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2  "+
            "If the question is about calculating the change between year averages, apply the previous logic and take difference. " +            
            "Expenses are revenue minus net income. " +            
            #"To calculate the difference, use absolute value. " +
            "To calculate difference, use equation: difference = abs(signed_new_value - signed_old_value)." +
            "To calculate change, use equation: change = (2*signed_new_value - 2*signed_old_value)/2. "+
            "To calculate percentage change, use equation: percentage_change = (2*signed_new_value - 2*signed_old_value)/2*signed_old_value * 100. " +
            "To calculate proportion, do not calculate percentage. " +
            "Use all given year values if no year specified. " +
            "The code must be specific to the provided value list. " +
            "Do not generate explanation, nor example code, just the function. "  ),
        (
          "ai",
          "Ok, I have all the information. The Python function is as follows:"
        ),
      ],

    
    'V6' : [#	0.625, full: 0.5846774193548387, 0.7862903225806451			
         ("system","You are a helpful assistant with a subtask in a question-answering pipeline. The questions are related to a financial report. " + 
             "Financial report is stored as a list of annotated values. You will receive the financial report as an annotated value list and the question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "Here is the financial report as a list of annotated values: {value_list}"
        ),         
        (
          "ai",
          "Ok, I received the value list."
        ),
        (
          "human",
          "Here is the question requires calculation on the financial report: {question}"
        ),
        (
          "ai",
          "Ok, I received the question."
        ),
        (
          "human",
            "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
            "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+
            "The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2  "+
            "If the question is about calculating the change between year averages, apply the previous logic and take difference. " +            
            "Expenses are revenue minus net income. " +            
            "To calculate the difference, use absolute value. " +
            #"To calculate difference, use equation: difference = abs(signed_new_value - signed_old_value)." +
            "To calculate change, use equation: change = (2*signed_new_value - 2*signed_old_value)/2. "+
            "To calculate percentage change, use equation: percentage_change = (2*signed_new_value - 2*signed_old_value)/2*signed_old_value * 100. " +
            "To calculate proportion, do NOT multiply by 100. " +
            "Use all given year values if no year specified. " +
            "The code must be specific to the provided value list. " +
            "Do not generate explanation, nor example code, just the function. "           
        ),
        (
          "ai",
          "Ok, I have all the information. The Python function is as follows:"
        ),
      ],
    'V7' : [
         ("system","You are a helpful assistant with a subtask in a question-answering pipeline. The questions are related to a financial report. " + 
             "Financial report is stored as a list of annotated values. You will receive the financial report as an annotated value list and the question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "Here is the financial report as a list of annotated values: {value_list}"
        ),         
        (
          "ai",
          "Ok, I received the value list."
        ),
        (
          "human",
          "Here is the question requires calculation on the financial report: {question}"
        ),
        (
          "ai",
          "Ok, I received the question."
        ),
        (
          "human",
            "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
            "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+
            "The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2  "+
            "If the question is about calculating the change between year averages, apply the previous logic and take difference. " +            
            "Expenses are revenue minus net income. " +            
            #"To calculate the difference, use absolute value. " +
            "To calculate difference, use equation: difference = abs(signed_new_value - signed_old_value)." +
            "To calculate change, use equation: change = (2*signed_new_value - 2*signed_old_value)/2. "+
            "To calculate percentage change, use equation: percentage_change = (2*signed_new_value - 2*signed_old_value)/2*signed_old_value * 100. " +
            "To calculate proportion, do NOT multiply by 100. " +
            "Use all given year values if no year specified. " +
            "The code must be specific to the provided value list. " +
            "Do not generate explanation, nor example code, just the function. "       ),
        (
          "ai",
          "Ok, I have all the information. The Python function is as follows:"
        ),
      ],

     'V8': [ # same as V2 except proportian calc
         ("system","You are a helpful assistant with a subtask in a question-answering pipeline. The questions are related to a financial report. " + 
             "Financial report is stored as a list of annotated values. You will receive the financial report as an annotated value list and the question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "Here is the financial report as a list of annotated values: {value_list}"
        ),         
        (
          "ai",
          "Ok, I received the value list."
        ),
        (
          "human",
          "Here is the question requires calculation on the financial report: {question}"
        ),
        (
          "ai",
          "Ok, I received the question."
        ),
        (
          "human",
            "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
            "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+
            "The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2  "+
            "If the question is about calculating the change between year averages, apply the previous logic and take difference. " +            
            "Expenses are revenue minus net income. " +            
            "To calculate the difference, use absolute value. " +
            #"To calculate difference, use equation: difference = abs(signed_new_value - signed_old_value)." +
            #"To calculate change, use equation: change = (2*signed_new_value - 2*signed_old_value)/2. "+
            "To calculate percentage change, use equation: percentage_change = (2*signed_new_value - 2*signed_old_value)/2*signed_old_value * 100. " +
            "To calculate proportion, do NOT multiply by 100. " +
            "Use all given year values if no year specified. " +
            "The code must be specific to the provided value list. " +
            "Do not generate explanation, nor example code, just the function. "           
        ),
        (
          "ai",
          "Ok, I have all the information. The Python function is as follows:"
        ),
      ],

     'V9': [
         ("system","You are a helpful assistant with a subtask in a question-answering pipeline. The questions are related to a financial report. " + 
             "Financial report is stored as a list of annotated values. You will receive the financial report as an annotated value list and the question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "Here is the financial report as a list of annotated values: {value_list}"
        ),         
        (
          "ai",
          "Ok, I received the value list."
        ),
        (
          "human",
          "Here is the question requires calculation on the financial report: {question}"
        ),
        (
          "ai",
          "Ok, I received the question."
        ),
        (
          "human",
            "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
            "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+
            "The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2  "+
            "If the question is about calculating the change between year averages, apply the previous logic and take difference. " +            
            "Expenses are revenue minus net income. " +            
            #"To calculate the difference, use absolute value. " +
            #"To calculate the difference, subtract from the bigger number. " +
            "To calculate difference, use equation: difference = abs(value1 - value2)." +
            "To calculate change, use equation: change = (2*signed_new_value - 2*signed_old_value)/2. "+
            #"To calculate change, use equation: change = (new_value - old_value)/2. "+
            "To calculate percentage change, use equation: percentage_change = (2*signed_new_value - 2*signed_old_value)/2*signed_old_value * 100. " +
            "To calculate proportion, do NOT multiply by 100. " +
            "Use all given year values if no year specified. " +
            "The code must be specific to the provided value list. " +
            "Do not generate explanation, nor example code, just the function. "           
        ),
        (
          "ai",
          "Ok, I have all the information. The Python function is as follows:"
        ),
      ],
    'V10': [
         ("system","You are a helpful assistant with a subtask in a question-answering pipeline. The questions are related to a financial report. " + 
             "Financial report is stored as a list of annotated values. You will receive the financial report as an annotated value list and the question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "Here is the financial report as a list of annotated values: {value_list}"
        ),         
        (
          "ai",
          "Ok, I received the value list."
        ),
        (
          "human",
          "Here is the question requires calculation on the financial report: {question}"
        ),
        (
          "ai",
          "Ok, I received the question."
        ),
        (
          "human",
            "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
            "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+
            "The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2  "+
            "If the question is about calculating the change between year averages, apply the previous logic and take difference. " +            
            "Expenses are revenue minus net income. " +            
            #"To calculate the difference, use absolute value. " +
            #"To calculate the difference, subtract from the bigger number. " +
            "To calculate difference, use equation: difference = abs(value1 - value2)." +
            #"To calculate change, use equation: change = (2*signed_new_value - 2*signed_old_value)/2. "+
            #"To calculate change, use equation: change = (new_value - old_value)/2. "+
            "To calculate percentage change, use equation: percentage_change = (2*signed_new_value - 2*signed_old_value)/2*signed_old_value * 100. " +
            "To calculate proportion, do NOT multiply by 100. " +
            "Use all given year values if no year specified. " +
            "The code must be specific to the provided value list. " +
            "Do not generate explanation, nor example code, just the function. "           
        ),
        (
          "ai",
          "Ok, I have all the information. The Python function is as follows:"
        ),
      ],

    'V11': [
         ("system","You are a helpful assistant with a subtask in a question-answering pipeline. The questions are related to a financial report. " + 
             "Financial report is stored as a list of annotated values. You will receive the financial report as an annotated value list and the question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "Here is the financial report as a list of annotated values: {value_list}"
        ),         
        (
          "ai",
          "Ok, I received the value list."
        ),
        (
          "human",
          "Here is the question requires calculation on the financial report: {question}"
        ),
        (
          "ai",
          "Ok, I received the question."
        ),
        (
          "human",
            "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
            "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+
            #"The calculation usually involves two steps: a selection and a calculation on selected data. "+
            #"If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2  "+
            #"If the question is about calculating the change between year averages, apply the previous logic and take difference. " +            
            #"Expenses are revenue minus net income. " +            
            #"To calculate the difference, use absolute value. " +
            #"To calculate the difference, subtract from the bigger number. " +
            #"To calculate difference, use equation: difference = abs(value1 - value2)." +
            #"To calculate change, use equation: change = (2*signed_new_value - 2*signed_old_value)/2. "+
            #"To calculate change, use equation: change = (new_value - old_value)/2. "+
            #"To calculate percentage change, use equation: percentage_change = (2*signed_new_value - 2*signed_old_value)/2*signed_old_value * 100. " +
            #"To calculate proportion, do NOT multiply by 100. " +
            #"Use all given year values if no year specified. " +
            #"The code must be specific to the provided value list. " +
            "Do not generate explanation, nor example code, just the function. "           
        ),
        (
          "ai",
          "Ok, I have all the information. The Python function is as follows:"
        ),
      ],

    'V12': [ ## table + code
         ("system","You are a helpful assistant with a subtask in a question-answering pipeline. The questions are related to a financial report. " + 
             "Financial report is stored as a table. You will receive the financial report as a table and the question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "Here is the financial report as a table: {table}"
        ),         
        (
          "ai",
          "Ok, I received the table."
        ),
        (
          "human",
          "Here is the question requires calculation on the financial report: {question}"
        ),
        (
          "ai",
          "Ok, I received the question."
        ),
        (
          "human",
            "Generate a Python function 'run(table)' that can answer the question using the table! "+
            "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+
            "The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2  "+
            "If the question is about calculating the change between year averages, apply the previous logic and take difference. " +            
            "Expenses are revenue minus net income. " +            
            "To calculate the difference, use absolute value. " +
            #"To calculate difference, use equation: difference = abs(signed_new_value - signed_old_value)." +
            #"To calculate change, use equation: change = (2*signed_new_value - 2*signed_old_value)/2. "+
            "To calculate percentage change, use equation: percentage_change = (2*signed_new_value - 2*signed_old_value)/2*signed_old_value * 100. " +
            "To calculate proportion, do not calculate percentage. " +
            "Use all given year values if no year specified. " +
            "The code must be specific to the provided table. " +
            "Do not generate explanation, nor example code, just the function. "              
        ),
        (
          "ai",
          "Ok, I have all the information. The Python function is as follows:"
        ),
      ],

    'V13': [ ## table + code - no rules
         ("system","You are a helpful assistant with a subtask in a question-answering pipeline. The questions are related to a financial report. " + 
             "Financial report is stored as a table. You will receive the financial report as a table and the question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "Here is the financial report as a table: {table}"
        ),         
        (
          "ai",
          "Ok, I received the table."
        ),
        (
          "human",
          "Here is the question requires calculation on the financial report: {question}"
        ),
        (
          "ai",
          "Ok, I received the question."
        ),
        (
          "human",
            "Generate a Python function 'run(table)' that can answer the question using the table! "+
            "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+
            #"The calculation usually involves two steps: a selection and a calculation on selected data. "+
            #"If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2  "+
            #"If the question is about calculating the change between year averages, apply the previous logic and take difference. " +            
            #"Expenses are revenue minus net income. " +            
            #"To calculate the difference, use absolute value. " +
            #"To calculate difference, use equation: difference = abs(signed_new_value - signed_old_value)." +
            #"To calculate change, use equation: change = (2*signed_new_value - 2*signed_old_value)/2. "+
            #"To calculate percentage change, use equation: percentage_change = (2*signed_new_value - 2*signed_old_value)/2*signed_old_value * 100. " +
            #"To calculate proportion, do not calculate percentage. " +
            #"Use all given year values if no year specified. " +
            #"The code must be specific to the provided table. " +
            "Do not generate explanation, nor example code, just the function. "              
        ),
        (
          "ai",
          "Ok, I have all the information. The Python function is as follows:"
        ),
      ],

    'V14': [ #no code + value list
         ("system","You are a helpful assistant in a question-answering pipeline. The questions are related to a financial report. " + 
             "Financial report is stored as a list of annotated values. You will receive the financial report as an annotated value list and the question. "+
             "Your task is to answer the received question. "
         ),
        (
          "human",
          "Here is the financial report as a list of annotated values: {value_list}"
        ),         
        (
          "ai",
          "Ok, I received the value list."
        ),
        (
          "human",
          "Here is the question requires calculation on the financial report: {question}"
        ),
        (
          "ai",
          "Ok, I received the question."
        ),
        (
          "human",
            "Format your answer as a tuple with format (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+            
            "Do not generate explanation, nor example code, just the tuple. "
        ),
        (
          "ai",
          "Ok, I have all the information. The result tuple is as follows:"
        ),
      ],  
     
    'V15': [ #no code + value list + rules
         ("system","You are a helpful assistant in a question-answering pipeline. The questions are related to a financial report. " + 
             "Financial report is stored as a list of annotated values. You will receive the financial report as an annotated value list and the question. "+
             "Your task is to answer the received question. "
         ),
        (
          "human",
          "Here is the financial report as a list of annotated values: {value_list}"
        ),         
        (
          "ai",
          "Ok, I received the value list."
        ),
        (
          "human",
          "Here is the question requires calculation on the financial report: {question}"
        ),
        (
          "ai",
          "Ok, I received the question."
        ),
        (
          "human",
            "Format your answer as a tuple with format (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+            
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2  "+
            "If the question is about calculating the change between year averages, apply the previous logic and take difference. " +            
            "Expenses are revenue minus net income. " +            
            "To calculate the difference, use absolute value. " +
            #"To calculate difference, use equation: difference = abs(signed_new_value - signed_old_value)." +
            #"To calculate change, use equation: change = (2*signed_new_value - 2*signed_old_value)/2. "+
            "To calculate percentage change, use equation: percentage_change = (2*signed_new_value - 2*signed_old_value)/2*signed_old_value * 100. " +
            "To calculate proportion, do not calculate percentage. " +
            "Use all given year values if no year specified. " +
            "Do not generate explanation, nor example code, just the tuple. "
        ),
        (
          "ai",
          "Ok, I have all the information. The result tuple is as follows:"
        ),
      ],  
}