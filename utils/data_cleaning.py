import json
from utils.load_and_save import save_to_json
import argparse
from config.config import Config

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="config_files/")
args = parser.parse_args()
cfg = Config(args.config)

def cleaning(data):
    """
    Cleans and processes a list of dictionaries containing objects and relations.

    This function extracts and filters relations and objects from the input data,
    maps object categories to macro-categories, and removes duplicates.

    **Processing Steps:**
    - Extracts two lists in the loop "for a in data", a can be or a Relation or a Object. The will have different keys
      - Relations keys (`src`, `dest`, `taxonomy_url`)
      - Objects keys (`label`, `uuid`, `taxonomy_url`)
    - Filters relations to keep only those in `ok_rel`.
    - Maps object categories to macro-categories using a predefined mapping.
    - Deduplicates objects and relations.
    - Saves the cleaned data to JSON files.

    **Duplicate Removal:**
    - Two dictionaries are considered duplicates if all their keys and values are identical.
    - Dictionaries are converted into ordered tuples to ensure consistent hashing.

    Args:
        data (list[dict]): 
            A list of dictionaries representing relations and objects.

    Returns:
        tuple[list[dict], list[dict]]: 
            - A list of unique objects with updated category mapping.
            - A list of unique relations filtered by allowed types.

    Saves:
        - `"data/obj_dict_filtered_final.json"`: The cleaned list of objects.
        - `"data/rel_dict_filtered_final.json"`: The cleaned list of relations.

    Example:
        ```python
        cleaned_objects, cleaned_relations = cleaning(data)
        ```
    """

    rel_dict = []
    obj_dict = []
    unknown = []

    for a in data:
        if "src" in a and "dest" in a and "taxonomy_url" in a:
            if len(a["src"])>0 and len(a["dest"])>0 and len(a["taxonomy_url"])>0:
                rel_dict.append({"src": a["src"], "dest": a["dest"], "taxonomy_url": a["taxonomy_url"]})
        elif "label" in a and "uuid" in a and "taxonomy_url" in a:
            if len(a["label"])>0 and  len(a["uuid"])>0 and  len(a["taxonomy_url"])>0:
                obj_dict.append({"label": a["label"], "uuid": a["uuid"], "taxonomy_url": a["taxonomy_url"]})
        else:
            unknown.append(a)
            
    #ok_rel = ['#hasProduction', '#hasEvent', '#hasOwnership', '#hasPublisher','#hasRecord', '#hasLegalDecision', '#hasActor']
    ok_rel = ["#IsRelated",'#hasProduction', '#hasEvent', '#hasOwnership', '#hasPublisher','#hasRecord', '#hasLegalDecision', '#hasActor']


    category_mapping_obj = {
        "#Person": "#Person",
        "#Groupofpersons":"#Organisations",
        "#Corporatebody":"#Organisations",
        "#Company": "#Organisations",
        "#School": "#Organisations",
        "#Government": "#Organisations",
        "#Organisation": "#Organisations",
        "#Museum": "#Organisations",
        "#Country": "#Locations",
        "#Region": "#Locations",
        "#City": "#Locations",
        "#Location": "#Locations",
        "#Production": "#Locations",
        "#Period": "#Time",
        "#Date": "#Time",
        "#Event": "#Event",
        "#Painting": "#Artifacts",
        "#Sculpture": "#Artifacts",
        "#Ceramics": "#Artifacts",
        "#Icon": "#Artifacts",
        "#Furniture": "#Artifacts",
        "#Jewellery": "#Artifacts",
        "#Glass": "#Artifacts",
        "#Textile": "#Artifacts",
        "#Wood": "#Artifacts",
        "#Metal": "#Artifacts",
        "#Money": "#Artifacts",
        "#Stone": "#Artifacts",
        "#Artifact": "#Artifacts",
        "#Vessels": "#Artifacts",
        "#MusicalInstruments": "#Artifacts",
        "#Books": "#Artifacts",
        "#Record": "#Artifacts",
        "#ReligiousAndRitualObject": "#Artifacts",
        "#PhysicalObject": "#Artifacts",
        "#Collection": "#Artifacts"
    }


 
    # OBJECT
    # obj_dict_filtered_final contains only #Person or #organisation in taxonomy_url
    #REMEMEBER:  Objects keys (`label`, `uuid`, `taxonomy_url`)
    obj_dict_filtered_final = []  
    for data in obj_dict:
        taxonomy = data["taxonomy_url"][0] # For semplicity only the first entry is considered,  in general it is a list
 
        if taxonomy in category_mapping_obj.keys():
            data["taxonomy_url"] =category_mapping_obj[taxonomy] # reassign without list format
            obj_dict_filtered_final.append(data)


    # RELATIONS
    # Filter only the relations of interest in ok-rel
    #REMEMEBER:  Relations keys (`src`, `dest`, `taxonomy_url`)
    rel_dict_filtered_final = []
    for data in rel_dict:
        taxonomy = data["taxonomy_url"][0] # For semplicity only the first entry is considered, in general it is a list

        if taxonomy in ok_rel:
            data["taxonomy_url"] = taxonomy # reassign without list format
            rel_dict_filtered_final.append(data)
              


        
    unique_obj_list_of_dict = []
    seen_obj = set()
    for d_obj in obj_dict_filtered_final:
        # Convert the dictionary into an ordered tuple (hashable)
        # Dictionaries are not hashable
        # They are sorted to maintain a consistent order
        t_obj = tuple(sorted(d_obj.items()))
        if t_obj not in seen_obj:
            seen_obj.add(t_obj)
            unique_obj_list_of_dict.append(d_obj)
 
    unique_rel_list_of_dict = []
    seen_rel = set()
    for d_rel in rel_dict_filtered_final:
        t_rel = tuple(sorted(d_rel.items()))
        if t_rel not in seen_rel:
            seen_rel.add(t_rel)
            unique_rel_list_of_dict.append(d_rel)


    uuid_taxonomy_obj_dict = {element["uuid"]: element["taxonomy_url"] for element in unique_obj_list_of_dict}
 
    return uuid_taxonomy_obj_dict, unique_obj_list_of_dict, unique_rel_list_of_dict


def remove_reciprocal_relations(triples):
    """
    Removes reciprocal relations from a list of triples and ensures the uniqueness of the remaining relations.
    
    A reciprocal relation is defined as a pair of triples where one is the inverse of the other, i.e., (a, b, c)
    and (c, b, a) are considered duplicates (undirected graph). Only the first occurrence of each unique relation (or its inverse) 
    is retained.

    Args:
        triples (list): A list of triples, where each triple is a list or tuple containing three elements:
                         - The first element represents the source.
                         - The second element represents the label or type of relationship.
                         - The third element represents the target.
    Returns:
        list: A filtered list of unique triples, where reciprocal relations are removed. 
          
    """

    seen = set()
    results=[]
    for triple in sorted(triples):

        t = tuple(triple)
        t_inv = (t[2], t[1], t[0])
        if t not in seen and t_inv not in seen:
            seen.add(t)
            results.append(list(t))
    return results
    




def post_proc(DIR,output_file,temp_output_path):
    """
    post process results of graph_proj_parallel.py and save them.
    In the post processing all the duplictaed triplets are removed as well as the reciprocal ones.
    if (a,b,c) and (d,e,f) are equal a=d, b=e, c=f
    the reciprocal of (a,b,c) is (c,b,a)

    Args:
        DIR (str): main data folder
        output_file (str): finel csv file with valid triplets to be used in link prediction
        final_results (list): list of results computed in graph_proj_parallel.py (triplets)
    """
    
    final_results = []
    with open(temp_output_path, 'r', encoding='utf-8') as f:
        for line in f:
            final_results.append(json.loads(line))
    print(f"Total length triplets BEFORE post processing {len(final_results)}")
    
    #file_path_output = DIR +"x4"+output_file
    #save_to_json(file_path_output,final_results)


        
    final_results = list(remove_reciprocal_relations(final_results)) 
    file_path_output = DIR + output_file

    print(f"Total length triplets AFTER post processing {len(final_results)} \n")

    save_to_json(file_path_output,final_results)
    
    print(f"File with generated triplets located in {DIR + cfg.DATA_FILENAME}\n")

 