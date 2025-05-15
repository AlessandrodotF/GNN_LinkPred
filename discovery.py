import torch
import pandas as pd
import os
from utils.dataset_utils import load_pd_from_json
from utils.load_and_save import save_to_json
import utils.dataset_utils as Dataset_utils
from torchkge.models import ComplExModel
from utils.discovery_functions import discover_facts


import argparse
from config.config import Config

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="config_files/")
args = parser.parse_args()
cfg = Config(args.config)


def discovery_func(params_GS):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df_full,df_train, df_test =  Dataset_utils.preprocess(directory_path=cfg.DIR,file_name=cfg.DATA_FILENAME,test_size=cfg.TEST_SIZE,val_size=cfg.VAL_SIZE)
    full_kg,train_kg,test_kg = Dataset_utils.get_KnowledgeGraph(df_full=df_full, df_train=df_train, df_test=df_test,df_val=None) 



    model = ComplExModel(params_GS["EMB_DIM"], train_kg.n_ent, train_kg.n_rel)
    model_path = os.path.join(cfg.LOAD_DIR_MODEL, cfg.MODEL_FILENAME)
    model.load_state_dict(torch.load(model_path, weights_only=True))  
    model.to(device)
    print("Modello caricato correttamente!")

    X = load_pd_from_json(cfg.DIR, cfg.DATA_FILENAME).astype(str)
    discovered_facts = discover_facts(X, model, full_kg, top_n=cfg.TOP_N, max_candidates=cfg.MAX_CANDIDATES, strategy=cfg.STRATEGY, target_rel=cfg.TARGET_REL, seed=cfg.SEED)


    df = pd.DataFrame(discovered_facts[0], columns=["from", "relation", "to"])
    df.to_json(cfg.DIR+"facts_torch_"+cfg.DATA_FILENAME,index=False, orient='records', lines=True)

    print(discovered_facts)