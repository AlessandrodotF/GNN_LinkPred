import json
from config.config import *
from utils.load_and_save import load_from_json,save_to_json


"""
This script was created to process the PROV_network dataset, which had a different structure compared to AAMD.
Specifically, the original file contained keys structured differently, as defined in the create_node function. 
This script reformats the data to match the AAMD format.

Additionally, it ensures data consistency by:
1. Removing duplicate entries.
2. Verifying that reciprocal relationships (A → B and B → A) are not present simultaneously.
3. Structuring the dataset into a list of unique nodes and relationships, maintaining consistency in taxonomy labels.

The final processed data is stored in a formatted JSON file. This will be given as input to graph_proj_parallel.py
"""



def create_node(data):
    """
    Creates a list of unique nodes from a list of dictionaries containing information about entities connected 
    by relationships. Each node is defined by the pair (label, uuid) and a formatted taxonomy_url.

    The function ensures that each node is unique by using a set to track nodes that have already been added. 
    Duplicates are avoided based on the combination of (source, source_id, source_type, target, target_id, target_type).

    Args:
        data (list): A list of dictionaries, where each dictionary represents a relationship between two entities.
                     Each dictionary must contain the following keys:
                     - "source": the source of the relationship (e.g., "Alice")
                     - "source_id": the ID of the source (e.g., 1) (int)
                     - "source_type": the type of the source entity (e.g., "Person")
                     - "target": the target of the relationship (e.g., "Bob")
                     - "target_id": the ID of the target (e.g., 2) (int)
                     - "target_type": the type of the target entity (e.g., "Person")

    Returns:
        list: A list of dictionaries representing the nodes. Each dictionary contains:
              - "label": the label of the node (either source or target) (as a string)
              - "uuid": the UUID of the node (as a string)
              - "taxonomy_url": a list containing a formatted_url (e.g., ["#Person"] in this specif format = 1 length list)

    """
    list_of_nodes = []
    seen = set()  # Set per tenere traccia dei nodi già inseriti (usando tuple come chiave)
    
    for element in data:
        for key in ("source", "target"):
            id_key = f"{key}_id"
            type_key = f"{key}_type"
            
            #check if the full triplet is valid ( = not None)
            if element.get(key) and element.get(id_key) and element.get(type_key): 
                key_tuple = (element[key], str(element[id_key]), element[type_key])
                
                if key_tuple not in seen:
                    seen.add(key_tuple)
                    #just a formatting option
                    taxonomy_url_value = f"#{element[type_key].replace(' ', '')}"

                    node = {
                        "label": element[key],
                        "uuid": str(element[id_key]),
                        "taxonomy_url": [taxonomy_url_value]
                    }
                    #append nodes not "seen"
                    list_of_nodes.append(node)
                    
    return list_of_nodes


def create_relation(data):
    """
    Creates a list of unique relationships between entities. Each relationship is defined by a source (src) 
    and a target (dest), and is associated with a fixed taxonomy URL indicating that the entities are related.

    The function ensures that each relationship is unique by using a set to track the (source, target) pairs that 
    have already been added. Duplicate relationships are avoided.

    Args:
        data (list): A list of dictionaries, where each dictionary represents a relationship between two entities.
                     Each dictionary must contain the following keys:
                     - "source_id": the ID of the source entity (e.g., 1) (int)
                     - "target_id": the ID of the target entity (e.g., 2) (int)

    Returns:
        list: A list of dictionaries representing the relationships. Each dictionary contains:
              - "src": the ID of the source entity (as a string)
              - "taxonomy_url": a list containing the relationship type (e.g., ["#IsRelated"] in this specif format = 1 length list)
              - "dest": the ID of the target entity (as a string)
    """
    list_of_relations = []
    seen=set()
  
    for element in data:
        source = str(element["source_id"])#source_id int in the original data
        target = str(element["target_id"])#target_id int in the original data
        rel = ["#IsRelated"] 
        key_tuple= (source,target)
        #ensure unique relations
        if key_tuple not in seen:
            dict_temporary = {"src": source, "taxonomy_url": rel, "dest": target}
            seen.add(key_tuple)
            list_of_relations.append(dict_temporary)
        
    return list_of_relations







def perform_preprocessing():
    
    input_path = DIR+WORKING_DATASET_FILE
    data = load_from_json(input_path)
    
    nodes = create_node(data) #list of dict with ONLY nodes
    rels = create_relation(data) #list of dict with ONLY relations
    
    #concatenation, WHY?
    # in the original AAMD dataset structure, both nodes and rels are present and concatenated in a single list of dictionaries
    final_dataset = nodes+rels

    save_path =  DIR+DATASET_NAME+"_clean.json"
    save_to_json(save_path,final_dataset)
