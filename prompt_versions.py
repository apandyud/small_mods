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
      'V16': [
         ("system","You will receive the financial report as an annotated value list and the question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, percent or an empty string. "+
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
        )        
      ],

    'V17': [
         ("system","You will receive the financial report as an annotated value list and the question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the QUESTION using the VALUE_LIST! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, percent or an empty string. "+
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
            "Let's think step-by-step"+
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],
     'V18': [
         ("system","You will receive the financial report as an annotated value list and the question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
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
        )        
      ],
    'V18r': [
         ("system","You will receive the financial report as an annotated value list and the question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string."+
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
        )        
      ],
    

          'V19': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+
            #"The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2  "+
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
        )        
      ],
      'V19r': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string."+
            #"The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2; "+
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
            #"never select by index;" +
            #"'percentage change' results 'percent' scale; " +
            #"'change in percentage' is a subtraction; " +
            #"To calculate the average, use all relevant years.;" +
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],
      'V20': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+
            #"The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2  "+
            "If the question is about calculating the change between year averages, apply the previous logic and take difference. " +            
            "Expenses are revenue minus net income. " +            
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
        )        
      ],
      'V20r': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string."+
            #"The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2; "+
            "If the question is about calculating the change between year averages, apply the previous logic and take difference.; " +            
            "Expenses are revenue minus net income.; " +            
            #"To calculate the difference, use absolute value. " +
            #"To calculate the difference, subtract from the bigger number. " +
            #"To calculate difference, use equation: difference = abs(value1 - value2)." +
            #"To calculate change, use equation: change = (2*signed_new_value - 2*signed_old_value)/2. "+
            #"To calculate change, use equation: change = (new_value - old_value)/2. "+
            #"To calculate percentage change, use equation: percentage_change = (2*signed_new_value - 2*signed_old_value)/2*signed_old_value * 100. " +
            #"To calculate proportion, do NOT multiply by 100. " +
            #"Use all given year values if no year specified. " +
            #"The code must be specific to the provided value list. " +
            #"never select by index;" +
            #"'percentage change' results 'percent' scale; " +
            #"'change in percentage' is a subtraction; " +
            #"To calculate the average, use all relevant years.;" +
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],
    'V21': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+
            #"The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2  "+
            "If the question is about calculating the change between year averages, apply the previous logic and take difference. " +            
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
        )        
      ],
      'V21r': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string."+
            #"The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2; "+
            "If the question is about calculating the change between year averages, apply the previous logic and take difference.; " +            
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
            #"never select by index;" +
            #"'percentage change' results 'percent' scale; " +
            #"'change in percentage' is a subtraction; " +
            #"To calculate the average, use all relevant years.;" +
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],
        'V22': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+
            #"The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2  "+
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
            "'percentage change' results 'percent' scale" +
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],
    'V22r': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string."+
            #"The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2; "+
            #"If the question is about calculating the change between year averages, apply the previous logic and take difference.; " +            
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
            #"never select by index;" +
            "'percentage change' results 'percent' scale; " +
            #"'change in percentage' is a subtraction; " +
            #"To calculate the average, use all relevant years.;" +
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],
        'V23': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
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
            "'percentage change' results 'percent' scale" +
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],
    'V23r': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string."+
            #"The calculation usually involves two steps: a selection and a calculation on selected data. "+
            #"If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2; "+
            #"If the question is about calculating the change between year averages, apply the previous logic and take difference.; " +            
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
            #"never select by index;" +
            "'percentage change' results 'percent' scale; " +
            #"'change in percentage' is a subtraction; " +
            #"To calculate the average, use all relevant years.;" +
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],
    'V24': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+
            #"The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2  "+
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
            "'percentage change' results 'percent' scale" +
            "'change in percentage' is a subtraction" +
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],
    'V24r': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string."+
            #"The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2; "+
            #"If the question is about calculating the change between year averages, apply the previous logic and take difference.; " +            
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
            #"never select by index;" +
            "'percentage change' results 'percent' scale; " +
            "'change in percentage' is a subtraction; " +
            #"To calculate the average, use all relevant years.;" +
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],
    #small refactor, adding ';' between rules  + GPT 5 recommended instr.   
    'V25': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string."+
            #"The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2; "+
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
            "never select by index;" +
            "'percentage change' results 'percent' scale; " +
            "'change in percentage' is a subtraction; " +
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],

    'V26': [
         ("system",
          "You will receive a financial report as an annotated value list and a question. "
    "Your task is to generate only a Python function that computes the numeric answer "
    "to the question from the list. "

    "Hard requirements (read carefully): "
    "1) Return format: the function must return a tuple (number, scale). "
    "   - number: float, rounded to two decimals. "
    "   - scale: one of '', 'thousand', 'million', 'billion', 'percent'. "
    "2) Selection discipline: never select by index; always by normalized header/entity/year from the question. "
    "   Prefer exact match; if multiple matches exist, choose the one with longest textual overlap and matching year/entity. "
    "   If no match, raise a clear exception (do not return NaN). "
    "3) Scale normalization: convert all inputs to base units before computing; convert to requested output only at the end. "
    "4) Percent / ratio semantics: "
    "   - ratio/proportion → 0–1; convert to percent only if explicitly requested. "
    "   - percentage change/YoY percent change → ((new-old)/old) * 100, scale='percent'. "
    "   - change in percentage → subtraction of percentages, scale='percent'. "
    "   - difference → absolute difference; change/increase/decrease → signed new-old. "
    "5) Signs: numbers shown in parentheses in financial reports are negative. "
    "6) Averages: filter out None/NaN; require at least 2 values; arithmetic mean unless question says weighted. "
    "7) Years and YoY: parse year(s) from question; for year on year: old=year-1, new=year. "
    "8) Defensive programming: "
    "   - On division check zero/None and raise error. "
    "   - Validate outputs: ratio in [0,1] unless percentage requested; percent typically in [0,100]. "
    "   - Round only at the very end. "

    "Generate only the function body requested in the user message. "
    "Do not include explanations, tests, prints, or example calls."
         ),
        (
          "human",
          "VALUE_LIST: {value_list}\n"
    "QUESTION: {question}\n"
    "Generate a Python function 'run(value_list)' that answers the question using the annotated values list. "
    "Implementation checklist: "
    "- Use helper utilities inside the function for normalization, scale handling, safe mean/division, and rounding. "
    "- Selection must not use numeric indexes; match by normalized header/entity/year from the question. "
    "- Semantics: "
    "   * Year average: average of year and year-1. "
    "   * percentage change/YoY percent change: ((new-old)/old)*100, scale='percent'. "
    "   * change in percentage: subtraction of percentages, scale='percent'. "
    "   * difference: absolute difference; change/increase/decrease: signed. "
    "- Parenthesized numbers represent negatives. "
    "- Return a tuple (number, scale). "
    "Output constraint: return only the function 'def run(value_list): ...' and nothing else."
        )       
    ],
    #to test with the two significant instructions
     'V27': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string."+
            #"The calculation usually involves two steps: a selection and a calculation on selected data. "+
            #"If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2; "+
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
            "never select by index;"+
            "'percentage change' results 'percent' scale; " +
            #"'change in percentage' is a subtraction; " +
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],
    #  v25 + recommended by qwen3 4b
    'V28': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string."+
            #"The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2; "+
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
            "When iterating over data, use appropriate methods like `for` loops or list comprehensions instead of `next()`;"+
            "never select by index;"+
            "'percentage change' results 'percent' scale; " +
            "'change in percentage' is a subtraction; " +
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],
    # same as v25, but without index rule, to see the effect of refactor
    'V29': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string."+
            #"The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2; "+
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
            #"never select by index;" +
            "'percentage change' results 'percent' scale; " +
            "'change in percentage' is a subtraction; " +
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],
     # same as v29, but without substract rule
    'V30': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string."+
            #"The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2; "+
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
            #"never select by index;" +
            "'percentage change' results 'percent' scale; " +
            #"'change in percentage' is a subtraction; " +
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],
     'V31': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string."+
            #"The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2; "+
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
            #"never select by index;" +
            "'percentage change' results 'percent' scale; " +
            "'change in percentage' is a subtraction; " +
            "To calculate the average, use all relevant years.;" +
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],
    'V32': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string."+
            #"The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2; "+
            "If the question is about calculating the change between year averages, apply the previous logic and take difference.; " +            
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
            #"never select by index;" +
            "'percentage change' results 'percent' scale; " +
            "'change in percentage' is a subtraction; " +
            #"To calculate the average, use all relevant years.;" +
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],

   
'V33': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string."+
            #"The calculation usually involves two steps: a selection and a calculation on selected data. "+
            #"If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2; "+
            #"If the question is about calculating the change between year averages, apply the previous logic and take difference.; " +            
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
            #"never select by index;" +
            #"'percentage change' results 'percent' scale; " +
            #"'change in percentage' is a subtraction; " +
            #"To calculate the average, use all relevant years.;" +
             '‘percentage change’ = (new-old)/old*100; ‘change in percentage’ = new% - old% (percentage points).\n' +
             'Year average = (value(y) + value(y-1)) / 2.\n' +
             'For ‘largest/smallest’, select the max/min after unit normalization; do not sum or average.\n' +
 'Clarify units/time window; ensure text-based column selection\n' +
 'Never index columns by position; select columns by header text (case-insensitive, stripped).\n' +
            
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],

   'V34': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
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
            "'percentage change' has formula '(a-b)/b' and results 'percent' scale;" +
            "'percentage' calculation has a formula 'a/b', with no multiplication and with empty scale;" +
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],

    'V34b': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
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
            "'percentage change' results 'percent' scale; " +
            "'percentage' has no multiplication and results empty scale; " +
            #"'change in percentage' is a subtraction; " +
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],
    'V35': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
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
            "'percentage change' has no multiplication  and results 'percent' scale; " +
            "'percentage' has no multiplication and results empty scale; " +
            "'change in percentage' is a subtraction and results 'percent' scale; " +
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],
     'V36': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
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
            "year average is the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2"
            "'percentage change' has no multiplication  and results 'percent' scale; " +
            "'percentage' has no multiplication and results empty scale; " +
            "'change in percentage' is a subtraction and results 'percent' scale; " +
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],
    'V36b': [
         ("system","You will receive the financial report as an annotated value list and a question. "+
             "Your task is to generate a Python function that can calculate a numeric value that is the answer for the received question. "                            
         ),
        (
          "human",
          "VALUE_LIST: {value_list}"+
          "QUESTION: {question}" +
          "Generate a Python function 'run(value_list)' that can answer the question using the list of annotated values! "+
          "The function must return a tuple (number, scale). The resulting number is a float with accuracy to two decimal places. Scale usually is thousand, million, billion, percent or an empty string. "+
            #"The calculation usually involves two steps: a selection and a calculation on selected data. "+
            "If the question is about calculating the year average, you must calculate the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2;"+
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
            #"year average is the average between the given year and the previous one. ex. 2015_average = (2015_value + 2014_value)/2"
            "'percentage change' has no multiplication  and results 'percent' scale; " +
            "'percentage' has no multiplication and results empty scale; " +
            "'change in percentage' is a subtraction and results 'percent' scale; " +
            "Do not generate explanation, nor example code, just the function. "           
        )        
      ],
}