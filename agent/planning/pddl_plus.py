'''
Utility file for handling PDDL+ domains and problems.

Using code from https://norvig.com/lispy.html by Peter Norvig
'''
'''
Return a numeric fluent from the list of fluents
'''
def get_numeric_fluent(fluent_list, fluent_name):
    for fluent in fluent_list:
        if fluent[0]=="=": # Note: I intentionally not added an AND with the next if, to not fall for cases where len(fluent)==1
            if fluents_names_equals(fluent[1], fluent_name):
                return fluent
    raise ValueError("Fluent %s not found in list" % fluent_name)

'''
Return the value of a given numeric fluent name
'''
def get_numeric_fluent_value(fluent):
    return fluent[-1]


'''
A fluent is defined by a identifier and a set of objects. 
Two fluent names are equal if these are equal.  
'''
def fluents_names_equals(fluent_name1, fluent_name2):
    if len(fluent_name1)!=len(fluent_name2):
        return False
    for i in range(len(fluent_name1)):
        if fluent_name1[i]!=fluent_name2[i]:
            return False
    return True


