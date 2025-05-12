import json


def load_from_json(path):           
    """
    Load from json file all the data

    Args:
        path (str): full file path where the data are stored

    Returns:
        list of dict: list of dict containing different information about src,dest ecc
    """
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_to_json(save_path,variable_to_save,indent=None):           
    """
    Save data in the specified path

    Args:
        path (str): full file path where the data are stored
        variable_to_save (Any): The variable (e.g., list, dict) to save.
        indent (int, optional): Indentation level for pretty-printing the JSON. Default is None (compact).

    """
    with open(save_path, 'w',encoding="utf-8") as json_file:
        json.dump(variable_to_save, json_file, indent=indent, ensure_ascii=False)      
        
        



