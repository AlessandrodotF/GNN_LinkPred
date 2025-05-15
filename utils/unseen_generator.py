import pandas as pd
import random
from utils.dataset_utils import load_pd_from_json
from utils.load_and_save import save_to_json



def swapping_tail(a_head,b_tail):
        

        swapped_tail = []
        a_b_H_T = [[elements[0],elements[1]] for elements in zip(a_head,b_tail)]
        b_a_T_H = [[elements[1],elements[0]] for elements in zip(a_head,b_tail)]
        b_tail_unique = list(sorted(set(b_tail)))
        
        for i in range(len(a_b_H_T)):
            random_element_tail = random.choice(b_tail_unique)  
            a_b_duplet = [a_b_H_T[i][0],random_element_tail] #ffirst trial duplet [H, Swapped Tail]
            
            flag= False
            while flag == False:
                if (a_b_duplet not in a_b_H_T) and (a_b_duplet not in b_a_T_H) and (a_b_duplet[0]!= a_b_duplet[1]): # check to decide the new tail
                    swapped_tail.append(random_element_tail)
                    flag=True #if true exit from while, new corrupted tail found
                  
                #if flag false continue the random tail sampling  
                random_element_tail = random.choice(b_tail_unique)  
                a_b_duplet = [a_b_H_T[i][0],random_element_tail]
                
        return pd.Series(swapped_tail)

def swapping_head(a_head,b_tail): #same of swappimg tail but for heads

        swapped_head = []
        a_b_H_T = [[elements[0],elements[1]] for elements in zip(a_head,b_tail)]
        b_a_T_H = [[elements[1],elements[0]] for elements in zip(a_head,b_tail)]
        a_head_unique = list(sorted(set(a_head)))
        
        for i in range(len(a_b_H_T)):
            random_element_head = random.choice(a_head_unique)  
            a_b_duplet = [random_element_head,a_b_H_T[i][1]]
            
            flag= False
            while flag == False:
                if a_b_duplet not in a_b_H_T and a_b_duplet not in b_a_T_H  and a_b_duplet[0]!= a_b_duplet[1]:
                    swapped_head.append(random_element_head)
                    flag=True
                random_element_head = random.choice(a_head_unique)  
                a_b_duplet = [random_element_head,a_b_H_T[i][1]]
            
        return pd.Series(swapped_head)



def generate_unseen_triplets_to_json(dir_path,data_filename,column_to_shaffle= "T"):

    

    print("Creating unseen triplets")
    df = load_pd_from_json(dir_path,data_filename)

    df = pd.DataFrame(df)
    head_col = df.iloc[:, 0]
    rel_col = df.iloc[:, 1]
    tail_col = df.iloc[:, 2]
    

    df = pd.concat([head_col.astype(str),rel_col.astype(str),tail_col.astype(str)], axis=1)
    df.columns = ["head", "relation", "tail"]  
    copy_df_rec = pd.concat([tail_col.astype(str),rel_col.astype(str),head_col.astype(str)], axis=1)
    copy_df_rec.columns = ["head", "relation", "tail"]

    
    
    if column_to_shaffle == "T":
        tail_col_swapped = swapping_tail(head_col,tail_col)
        new_df = pd.concat([head_col.astype(str),rel_col.astype(str),tail_col_swapped.astype(str)], axis=1)
        new_df.columns = ["head", "relation", "tail"]    
    elif column_to_shaffle == "H":
        head_col_swapped = swapping_head(head_col,tail_col)
        new_df = pd.concat([head_col_swapped.astype(str),rel_col.astype(str),tail_col.astype(str)], axis=1)
        new_df.columns = ["head", "relation", "tail"]    
    else:
        print("Head (H), Tail (T) only are swappable")
    
 

    data = new_df.values.tolist()

    json_file_path=dir_path + 'UNSEEN_' + data_filename
    save_to_json(json_file_path,data,indent=4)
    
    print(f"File with UNSEEN triplets saved and located in {json_file_path}")