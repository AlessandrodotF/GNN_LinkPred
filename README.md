## Main algorithm flow
**` Python 3.11.11 `**

This code tackles the task of **link prediction** using PyTorch Geometric and a scraped dataset saved as single list of dictionary in .json format.
- 0 Step: create the folder **`dataset`** and the corresponding subfolder with you dataset name.
- First step:  check if the file  **`DATA_FILENAME = f"{DATASET_NAME}_FINAL_valid_triplets.json"`** exists, if it exists, skip to the SECOND STEP otherwise it will be _created_. In any case, the _FINAL_valid_triplets file  will contain a set of triplets in the format **HEAD** **REL** **TAIL** (for example: `["liliappleton","#IsRelated","mother"] `). Detail about the _creation_:
    -  DATA_CLEAN (graph_parallel_proj.py) parameters  determines whether preprocess is needed. The idea is to transform PROV_network-like text structure to obtain the same present in AAMD. The result of the preprocess is saved in "{DATASET_NAME}_clean.json". If DATA_CLEAN is True this step is skipped.
    - Using **NetworkX** two independent dict are created storing the elementary graphs that grow from each single source and dest node in the dataset, only the closest (1-hop) neighbors are computed and connected. All the results are stored in **global_dict** variable and all the possible keys are stored in **global_valid_keys**
    - **build_graph_n_steps** takes a single elementary graph and fill it with all the possible related nodes, **compress_graph_nx** compress the graph maintaining only  the nodes connected through a link corresponding to the TAXONOMY_OF_INTEREST field.
    - This is repeated for all the elementary graph in **global_dict** in parallel (in parallel using CPU workers) and at each batch, results are saved in a jsonl file to optimize the usage of RAM
    - The final **post_proc** function loads the temporary file produced in .jsonl format, cleans the data from duplicates and removes the reciprocal relation.The result is saved in "{DATASET_NAME}_FINAL_valid_triplets.json"

- Second step: The**`DATA_FILENAME = f"{DATASET_NAME}_FINAL_valid_triplets.json"`** will be loaded and from its length, the fraction of samples to be used for test and val phases will be inferred
- Third step: 4 Pandas DataFrame will be generated from .json file. Three of them are for the train, validation, and test splits, and there is also a global version including all the data.
- Fourth step: The corresponding **`KnowledgeGraph`** will be computed using the PyTorch Geometric library
- Fifth step: **`Gridsearch`** will be performed and the parameters will be saved to find the best set of parameters
- Sixth step: the **`best parameters`** will be loaded to perform the **`training`** procedure for more epochs
- Seventh step: Inference on the **`Unseen`** set of triplets. If not present it will be generated according to the original dataset swapping Head OR Tail.



#### Bonus:
If you want to experiment with your own dataset:

- Create a folder in `dataset/` with the name of your custom data.
  (use **`DATASET_NAME = "custom_data"`**), do the same in  
  **`LOAD_DIR_MODEL = f"Saved_Models/ComplEx/{DATASET_NAME}/"`**  
  to access the data and save/load the associated model.

- If the file you start from is cleaned (like `AAMD_CLEAN.JSON`) use:  
    **`WORKING_DATASET_FILE = f"{DATASET_NAME}_clean.json"`**  
  (_in this way no preprocess will happen_)  
  otherwise:  
    **`WORKING_DATASET_FILE = f"{DATASET_NAME}_no_clean.json"`**  
  (_the preprocessing will generate the corresponding_  
   **`f"{DATASET_NAME}_clean.json"`** _in_ **`DIR = f"dataset/{DATASET_NAME}/"`**)

- Select **`TAXONOMY_OF_INTEREST`** keeping the list format.  
  Possible options are limited to:  
  `['#Person', '#Organisations', '#Locations', '#Time', '#Event', '#Artifacts']`  
  This will be used to compress the networks in `graph_proj_parallel.py`.

- With all the passages above you should have obtained a file called:  
    **`DATA_FILENAME = f"{DATASET_NAME}_FINAL_valid_triplets.json"`**  
  to train the GNN score model.

- Use:  
    **`ADD_RECIPROCAL_RELS = True`**  
  if you want to consider reciprocal relation (both on training and inference after the split of train/val/test).

- The gridsearch should run without problems as well as the training phase  
  if the folder **`LOAD_DIR_MODEL`** is correctly created.

- In the inference phase: If you have a file with unseen triplets for inference,  
  ensure that it is named:  
    **`UNSEEN_TRIPLES = f"UNSEEN_{DATA_FILENAME}"`**  
  and it is in the correct format both in structure and name.  
  If it is not present, it will be created accordingly swapping by default the **TAIL**.  
  In `test_and_inference_ComplEx.py` you can also swap **TAIL** or **HEAD**,  
  the relation will be untouched (in current version).

- The final output will be:  
    **`SAVE_TRIPLETS_SCORES = "statements_and_metrics_" + UNSEEN_TRIPLES[:-5] + ".csv"`**  
  where you will have all the metrics (**score**, **rank**, **prob**) for each unseen triplet in CSV format.

