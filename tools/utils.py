"""Collection of general-purpose routines."""

def process_param_string(param_string):
    """
    Parse a string of parameter name/value pairs into a dict.
    
    The input string is assumed to be of the format
    'name1: value1, name2: value2, ...'. It will be converted to a dict
    with keys name1, name2, ... and associated values value1, value2, ...

    """
    # Deal with None or empty input -- empty dict
    if param_string is None:
        return {}
    if ':' not in param_string:
        return {}

    param_list = param_string.split(',')

    param_dict = {}
    for param in param_list:
        elements = param.split(':')

        # Remove possible white space around separators and dictify
        name = elements[0].strip()
        value = elements[1].strip()

        # Attempt to convert float and int values to numbers
        try:
            value_num = int(value)
        except ValueError:
            
            try:
                value_num = float(value)
            except ValueError:
                value_num = value
            
        param_dict[name] = value_num

    return param_dict
