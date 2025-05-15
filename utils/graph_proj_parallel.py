from config.config import *
import time
import os
import json
from multiprocessing import Pool, cpu_count
from utils.data_cleaning import cleaning
from utils.graphs_utils import build_atomic_graphs, compress_graph_nx, build_graph_n_steps, build_graph_networkx
from utils.load_and_save import load_from_json
from utils.pre_preprocessing import perform_preprocessing
import argparse
from config.config import Config

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="config_files/")
args = parser.parse_args()
cfg = Config(args.config)



def init_worker(dict_graph_src, dict_graph_dest, uuid_taxonomy_obj_dict):
    """
    Init shareb global variables in the worker process
    """
    
    global _shared_dict_graph_src, _shared_dict_graph_dest, _shared_uuid_taxonomy_obj_dict
    _shared_dict_graph_src = dict_graph_src
    _shared_dict_graph_dest = dict_graph_dest
    _shared_uuid_taxonomy_obj_dict = uuid_taxonomy_obj_dict


def process_key(args):
    """
    function used by each worker
    """
    selected_category, selected_key = args
    step_n = build_graph_n_steps(
        _shared_dict_graph_src,
        _shared_dict_graph_dest,
        selected_category,
        selected_key,
        _shared_uuid_taxonomy_obj_dict
    )
    comp = compress_graph_nx(step_n, cfg.TAXONOMY_OF_INTEREST)
    edges = [[min(u,v), info['label'], max(u,v)] for u, v, info in comp.edges(data=True)]
    return edges

def process_and_save_batch(task_list, batch_size, pool, output_path):
    """
    Perform tasks in batch and save results of each batch in a jsonl file (optimeze the ram usage)
    Ensure JUST uniqueness!!!
    """
    batch_results = set()
    with open(output_path, 'a', encoding='utf-8') as fout:
        
        for i, result in enumerate(pool.imap(process_key, task_list, chunksize=int(cfg.BATCH_SIZE_PARALLEL*0.1))):
            
            batch_results.update(tuple(x) for x in result)
            
            if (i + 1) % batch_size == 0 or i == len(task_list) - 1:

                print(f"ITERATION {i+1}/{len(task_list)}")
                for triplet in batch_results:
                    fout.write(json.dumps(triplet, ensure_ascii=False) + '\n')
                    
                batch_results.clear()



def from_texts_to_triplets():
    print(f"Creation of the graphs and triplets for {cfg.DATASET_NAME}\n")

    # load or preprocess
    if "_no_clean.json" in  cfg.WORKING_DATASET_FILE:
        print("Data cleaning needed","\n")
        perform_preprocessing()#create a dataset with the same structure of AAMD and save results with _clean suffix
        data = load_from_json(cfg.DIR + cfg.DATASET_NAME + '_clean.json')
    else:
        print("Data cleaning NOT needed","\n")

        data = load_from_json(cfg.DIR + cfg.WORKING_DATASET_FILE) #your data have the same structure of AAMD e the corresponding file already have _clean suffix



    # Cleaning e construction of dicts 
    uuid_taxonomy_obj_dict, obj_dict_filtered_final, rel_dict_filtered_final = cleaning(data)
    global_dict, global_valid_keys = build_atomic_graphs(rel_dict_filtered_final, obj_dict_filtered_final)

    # graphs NetworkX
    dict_graph_src = {key: build_graph_networkx(global_dict['src_dict'], key) for key in global_valid_keys['src_dict']}
    dict_graph_dest = {key: build_graph_networkx(global_dict['dest_dict'], key) for key in global_valid_keys['dest_dict']}

    # task_list for the parallel execution  (category, key)
    task_list = [
        (category, key)
        for category in global_valid_keys
        for key in global_valid_keys[category]
    ]

    # path  output JSONL: remove if it exists (dengerous overwrite stuff), need to store non post processed results
    temp_output_path = cfg.DIR + 'triplets_temp.jsonl'
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    # Pool with initializer to share common variables between workers
    T1 = time.time()
    with Pool(
        processes=cpu_count(),
        initializer=init_worker,
        initargs=(dict_graph_src, dict_graph_dest, uuid_taxonomy_obj_dict)
    ) as pool:
        process_and_save_batch(task_list, cfg.BATCH_SIZE_PARALLEL, pool, temp_output_path)
    T2 = time.time()



    print(f"SAVED. Elapsed time = {T2 - T1:.2f} sec")
    return temp_output_path #gives in output the path where results are stored (neede to be post_processed)