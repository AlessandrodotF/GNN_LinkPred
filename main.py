#%%
import os
import utils.dataset_utils as Dataset_utils

from utils.graph_proj_parallel import from_texts_to_triplets
from complEx_GridSearch import grid_search_and_save_params
from train_ComplEx import full_training
from test_and_inference_ComplEx import inference
from utils.load_and_save import load_from_json
from utils.data_cleaning import post_proc
import utils.set_seeds_all as set_seeds_all 
import argparse
from config.config import Config

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="config_files_data/")
args = parser.parse_args()
cfg = Config(args.config)



#%%
# Load and preprocess data, from texts to triplets
if __name__ == '__main__':
    set_seeds_all.set_seed(cfg.SEED)
    
    if os.path.exists(cfg.DIR+cfg.DATA_FILENAME) == False:
        temp_output_path = from_texts_to_triplets() #store the intermediate results (help the memory managment)
        # handle preprocessed and not preprocessed data
        post_proc(cfg.DIR, cfg.DATA_FILENAME,temp_output_path)
#
    #split in train,val,test 
    #from this point DATA_FILENAME contains the triplets of interest
    X = Dataset_utils.load_pd_from_json(cfg.DIR, cfg.DATA_FILENAME,cfg.ADD_RECIPROCAL_RELS).astype(str)
    factor = 2 if cfg.ADD_RECIPROCAL_RELS else 1
    tot_len_data = factor*len(X)
    n_val= int(tot_len_data*cfg.VAL_SIZE)
    n_test= int(tot_len_data*cfg.TEST_SIZE)
    df_full, df_train, df_val, df_test =  Dataset_utils.preprocess(directory_path=cfg.DIR,file_name=cfg.DATA_FILENAME,test_size=n_test,val_size=n_val,add_reciproc=cfg.ADD_RECIPROCAL_RELS)
    print("=" * 40)
    print(f"Full len {len(df_full)}")
    print(f"Train len {len(df_train)}")
    print(f"Val len {len(df_val)}")
    print(f"Test len {len(df_test)}")
    print("=" * 40)
    full_kg,train_kg,val_kg,test_kg = Dataset_utils.get_KnowledgeGraph(df_full=df_full, df_train=df_train, df_test=df_test,df_val=df_val) 
    #needed to find the best iperparams (saved in .json file )
    grid_search_and_save_params(train_kg,val_kg)
    #load results from grid search (params_GS is a dict)
    params_GS = load_from_json(cfg.LOAD_DIR_MODEL + cfg.DATA_FILENAME[:-5]+"_best_params.json")
    full_training(train_kg, params_GS)
    #test(train_kg, test_kg, params_GS)
    inference(full_kg, train_kg, params_GS)
    
# %%