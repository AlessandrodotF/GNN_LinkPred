import torch
import numpy as np
import pandas as pd
import os
import utils.unseen_generator as unseen_generator
import utils.dataset_utils as Dataset_utils
from scipy.special import expit
from torchkge.evaluation import LinkPredictionEvaluator
from torchkge.models import ComplExModel
from config.config import *


def test(train_kg, test_kg, params_GS):

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device =  'cpu'

    model = ComplExModel(params_GS["EMB_DIM"], train_kg.n_ent, train_kg.n_rel)
    model_path = os.path.join(LOAD_DIR_MODEL, MODEL_FILENAME)
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 
    model.to(device)
    print("-"*10,"\n")
    print("Test SCORES","\n")
    print("-"*10,"\n")
    with torch.no_grad():
        evaluator = LinkPredictionEvaluator(model, test_kg)
        evaluator.evaluate(b_size=BATCH_SIZE, verbose=False)
        evaluator.print_results(k=[10,3,1], n_digits=2)




def inference(full_kg, train_kg, params_GS):

    flag = Dataset_utils.check_for_unseen_file(DIR,UNSEEN_TRIPLES)
    if flag: 
        X_unseen,unseen_kg = Dataset_utils.preprocess_unseen(directory_path=DIR,file_name=UNSEEN_TRIPLES,full_kg=full_kg)
    else:
        unseen_generator.generate_unseen_triplets_to_json(dir_path=DIR,data_filename=DATA_FILENAME,column_to_shaffle= "T")
        X_unseen,unseen_kg = Dataset_utils.preprocess_unseen(directory_path=DIR,file_name=UNSEEN_TRIPLES,full_kg=full_kg)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ComplExModel(params_GS["EMB_DIM"], train_kg.n_ent, train_kg.n_rel)
    model_path = os.path.join(LOAD_DIR_MODEL, MODEL_FILENAME)
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 
    model.to(device)
    
    

    model.eval()
    with torch.no_grad():
        evaluator = LinkPredictionEvaluator(model, unseen_kg)
        evaluator.evaluate(b_size=1, verbose=True)
        ranks_f_unseen_heads = evaluator.filt_rank_true_heads
        ranks_f_unseen_tails = evaluator.filt_rank_true_tails
        avg_ranks_filtered= torch.stack([ranks_f_unseen_heads.float(), ranks_f_unseen_tails.float()], dim=1).mean(dim=1)


        h_idx_tensor = unseen_kg.head_idx.clone().detach().to(device)
        t_idx_tensor = unseen_kg.tail_idx.clone().detach().to(device)
        r_idx_tensor = unseen_kg.relations.clone().detach().to(device)

        scores = model.scoring_function(h_idx_tensor, t_idx_tensor, r_idx_tensor)


    statements_and_metrics = pd.DataFrame(list(zip([' '.join(x) for x in X_unseen], 
                          avg_ranks_filtered.cpu().numpy().astype(int), 
                          np.squeeze(scores.cpu().numpy()),
                          np.squeeze(expit(scores.cpu().numpy())))), 
                 columns=['statement(H,T,R)', 'rank', 'score', 'prob'])


    #formatting options for terminal
    pd.set_option('display.max_columns', None)  
    pd.set_option('display.width', None)  
    pd.set_option('display.max_rows', None)  
    pd.set_option('display.max_colwidth', None)  




    statements_and_metrics = statements_and_metrics.sort_values("score", ascending=False)
    statements_and_metrics['score'] = statements_and_metrics['score'].apply(lambda x: f"{x:.3g}")
    statements_and_metrics['prob'] = statements_and_metrics['prob'].apply(lambda x: f"{x:.3g}")
    print("\n")
    print("-"*10,"\n")
    statements_and_metrics = statements_and_metrics.drop_duplicates().dropna().reset_index(drop=True)
    statements_and_metrics.to_csv(DIR+SAVE_TRIPLETS_SCORES, index=False, sep=",")
    print(f"Final statements and metrics saved in {DIR+SAVE_TRIPLETS_SCORES}")

