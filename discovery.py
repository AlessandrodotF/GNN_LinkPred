import torch
import pandas as pd
import os
from utils.dataset_utils import load_pd_from_json
from utils.load_and_save import save_to_json
import utils.dataset_utils as Dataset_utils
from torchkge.models import ComplExModel
from utils.discovery_functions import discover_facts
from config.config import *




def discovery_func(params_GS):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df_full,df_train, df_test =  Dataset_utils.preprocess(directory_path=DIR,file_name=DATA_FILENAME,test_size=TEST_SIZE,val_size=VAL_SIZE)
    full_kg,train_kg,test_kg = Dataset_utils.get_KnowledgeGraph(df_full=df_full, df_train=df_train, df_test=df_test,df_val=None) 



    model = ComplExModel(params_GS["EMB_DIM"], train_kg.n_ent, train_kg.n_rel)
    model_path = os.path.join(LOAD_DIR_MODEL, MODEL_FILENAME)
    model.load_state_dict(torch.load(model_path, weights_only=True))  
    model.to(device)
    print("Modello caricato correttamente!")

    X = load_pd_from_json(DIR, DATA_FILENAME).astype(str)
    discovered_facts = discover_facts(X, model, full_kg, top_n=TOP_N, max_candidates=MAX_CANDIDATES, strategy=STRATEGY, target_rel=TARGET_REL, seed=SEED)


    df = pd.DataFrame(discovered_facts[0], columns=["from", "relation", "to"])
    df.to_json(DIR+"facts_torch_"+DATA_FILENAME,index=False, orient='records', lines=True)

    print(discovered_facts)