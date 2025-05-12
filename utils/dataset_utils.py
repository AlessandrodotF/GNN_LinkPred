import os
import numpy as np
import pandas as pd
from torchkge.data_structures import KnowledgeGraph
from config.config import *

import os



def load_pd_from_json(directory_path, file_name, add_reciprocal_rels=False):
    """
    Load a NumPy array of triples from a JSON file.

    Each element in the JSON file should be a list representing a triple with subject, predicate, and object.
    The JSON file should have an array of arrays, where each inner array contains exactly three elements: the subject, predicate, and object.
    For example, the JSON file might look like this:

    .. code-block:: json

       [
           ["subj1", "#relationX", "obj1"],
           ["subj1", "#relationY", "obj2"],
           ["subj3", "#relationZ", "obj2"],
           ["subj4", "#relationY", "obj2"],
           ...
       ]

    Parameters
    ----------
    directory_path : str
        The folder where the JSON file is located.
    file_name : str
        The name of the JSON file.
    add_reciprocal_rels : bool, optional
        If True, for every triple ``<s, p, o>`` a reciprocal triple ``<o, p_reciprocal, s>`` will be added (default: False).

    Returns
    -------
    ndarray of shape (n, 3)
        An array containing the triples loaded from the file.
    """
    df = pd.read_json(
        os.path.join(directory_path, file_name),
        dtype=str,
        encoding="utf-8",
    )
    df = df.drop_duplicates()
    if add_reciprocal_rels:
        df = _add_reciprocal_relations(df)

    return df.values



def _add_reciprocal_relations(triples_df):
    """
    Add reciprocal relations to a DataFrame of triples.

    For each triple in the provided DataFrame, this function creates a reciprocal triple by
    swapping the subject and object.
    This version does not modify the predicate of the reciprocal triples (unlike previous versions that appended '_reciprocal').
    The reciprocal triples are then concatenated with the original triples.

    Parameters
    ----------
    triples_df : pandas.DataFrame
        A DataFrame containing triples in the format [subject, predicate, object] equivalently [head, rel, tail].

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing both the original and reciprocal triples.
    """
    # create a copy of the original triples to add reciprocal relations
    col1 = triples_df[0]
    col2 = triples_df[1]
    col3 = triples_df[2]
    
    swapp_df = pd.DataFrame([col3, col2, col1]).T
    swapp_df.columns = triples_df.columns  # Assicura che le colonne abbiano gli stessi nomi

    triples_df = pd.concat([triples_df, swapp_df])

    return triples_df

def filter_unseen_entities(X, train_kg):
    """
    Filter out triples that contain entities not present in the training knowledge graph.

    This function removes any triple from the input array if either its subject or object
    does not exist in the training knowledge graph's entity mapping.

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        An array of triples in the format [subject, predicate, object].
    train_kg : KnowledgeGraph
        The knowledge graph used during training, which defines the set of known entities.

    Returns
    -------
    ndarray of shape (m, 3)
        An array of triples where every entity is present in the training knowledge graph.
    """
    
    ent_seen = np.unique(list(train_kg.ent2ix.keys()))
    df = pd.DataFrame(X, columns=["s", "p", "o"])
    filtered_df = df[df.s.isin(ent_seen) & df.o.isin(ent_seen)]
    n_removed_ents = df.shape[0] - filtered_df.shape[0]
    if n_removed_ents > 0:
        return filtered_df.values
    return X


def train_test_split_no_unseen(
    X,
    test_size=100,
    allow_duplication=False,
    filtered_test_predicates=None,
):
    """
    Split the dataset into training and test sets while ensuring no unseen entities.

    This function creates a test set such that all entities and relations in the test set
    also appear in the training set. It iteratively selects candidate triples for the test set,
    verifying that their removal does not cause any entity or relation to appear only once.
    If necessary, and if allowed, duplicate triples may be added to meet the desired test set size.

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        The array of triples to split, in the format [subject, predicate, object].
    test_size : int or float, optional
        If an int, the number of triples to include in the test set.
        If a float, the proportion of the candidate triples to include (default: 100).

    allow_duplication : bool, optional
        If True, allows duplicated triples in the test set when it is not possible
        to obtain the required number of unique triples without introducing unseen entities.
    filtered_test_predicates : None or list, optional
        If provided, only triples with a predicate in this list are considered for the test set.
        If None, all predicates are considered.

    Returns
    -------
    X_train : ndarray of shape (m, 3)
        The training set of triples.
    X_test : ndarray of shape (k, 3)
        The test set of triples.

    Raises
    ------
    Exception
        If it is not possible to create a test set of the desired size without causing unseen entities,
        and duplication is not allowed.
    """


    if filtered_test_predicates:
        candidate_idx = np.isin(X[:, 1], filtered_test_predicates)
        X_test_candidates = X[candidate_idx]
        X_train = X[~candidate_idx]
    else:
        X_train = None
        X_test_candidates = X

    if isinstance(test_size, float):
        test_size = int(len(X_test_candidates) * test_size)

    entities, entity_cnt = np.unique(
        np.concatenate([X_test_candidates[:, 0], X_test_candidates[:, 2]]),
        return_counts=True,
    )
    rels, rels_cnt = np.unique(X_test_candidates[:, 1], return_counts=True)
    dict_entities = dict(zip(entities, entity_cnt))
    dict_rels = dict(zip(rels, rels_cnt))
    idx_test = []
    idx_train = []

    all_indices_shuffled = np.random.permutation(
        np.arange(X_test_candidates.shape[0])
    )

    for i, idx in enumerate(all_indices_shuffled):
        test_triple = X_test_candidates[idx]
        # reduce the entity and rel count
        dict_entities[test_triple[0]] = dict_entities[test_triple[0]] - 1
        dict_rels[test_triple[1]] = dict_rels[test_triple[1]] - 1
        dict_entities[test_triple[2]] = dict_entities[test_triple[2]] - 1

        # test if the counts are > 0
        if (
            dict_entities[test_triple[0]] > 0
            and dict_rels[test_triple[1]] > 0
            and dict_entities[test_triple[2]] > 0
        ):
            # Can safetly add the triple to test set
            idx_test.append(idx)
            if len(idx_test) == test_size:
                # Since we found the requested test set of given size
                # add all the remaining indices of candidates to training set
                idx_train.extend(list(all_indices_shuffled[i + 1:]))

                # break out of the loop
                break

        else:
            # since removing this triple results in unseen entities, add it to
            # training
            dict_entities[test_triple[0]] = dict_entities[test_triple[0]] + 1
            dict_rels[test_triple[1]] = dict_rels[test_triple[1]] + 1
            dict_entities[test_triple[2]] = dict_entities[test_triple[2]] + 1
            idx_train.append(idx)

    if len(idx_test) != test_size:
        # if we cannot get the test set of required size that means we cannot get unique triples
        # in the test set without creating unseen entities
        if allow_duplication:
            # if duplication is allowed, randomly choose from the existing test
            # set and create duplicates
            duplicate_idx = np.random.choice(
                idx_test, size=(test_size - len(idx_test))
            ).tolist()
            idx_test.extend(list(duplicate_idx))
        else:
            # throw an exception since we cannot get unique triples in the test set without creating
            # unseen entities
            raise Exception(
                "Cannot create a test split of the desired size. "
                "Some entities will not occur in both training and test set. "
                "Set allow_duplication=True,"
                "remove filter on test predicates or "
                "set test_size to a smaller value."
            )

    if X_train is None:
        X_train = X_test_candidates[idx_train]
    else:
        X_train_subset = X_test_candidates[idx_train]
        X_train = np.concatenate([X_train, X_train_subset])
    X_test = X_test_candidates[idx_test]

    X_train = np.random.permutation(X_train)
    X_test = np.random.permutation(X_test)

    return X_train, X_test


def preprocess(directory_path,file_name,test_size=100,val_size=0,add_reciproc=False):
    """
    Load triples from a JSON file and split them into training, validation, and test DataFrames in the [from, to, rel] format.

    This function reads a JSON file containing triples, splits the data into training and test sets,
    and optionally further splits the training set into a validation set. Note that if `val_size` is 0,
    the function returns only the full, training, and test DataFrames.

    Parameters
    ----------
    directory_path : str
        The folder where the JSON file is located.
    file_name : str
        The name of the JSON file.
    test_size : int or float, optional
        If an int, the number of triples to include in the test set.
        If a float, the proportion of triples to include in the test set.
    val_size : int or float, optional
        If an int, the number of triples to include in the validation set.
        If a float, the proportion of the training triples to allocate as validation.

    Returns
    -------

    df_full : pandas.DataFrame
        DataFrame containing the full set of triples (concatenation of training and test sets).
    df_train : pandas.DataFrame
        DataFrame containing the training triples in the [from, to, rel] format.
    df_val : pandas.DataFrame
        DataFrame containing the validation triples in the [from, to, rel] format.
    df_test : pandas.DataFrame
        DataFrame containing the test triples in the [from, to, rel] format.
    """
    X = load_pd_from_json(directory_path, file_name,add_reciproc).astype(str)

    if val_size == 0:
    
        X_train, X_test = train_test_split_no_unseen(X, test_size=test_size) 
        X_train_reordered = X_train[:, [0, 2, 1]]  #swap column in ['from', 'to', 'rel'] format
        X_test_reordered = X_test[:, [0, 2, 1]] #swap column in ['from', 'to', 'rel'] format
        X_full = np.concatenate([X_train_reordered,X_test_reordered])

        df_full=pd.DataFrame(X_full, columns=['from', 'to', 'rel'])
        df_train = pd.DataFrame(X_train_reordered, columns=['from', 'to', 'rel'])
        df_test = pd.DataFrame(X_test_reordered, columns=['from', 'to', 'rel'])

        return df_full,df_train, df_test
    
    else:
        X_train_intermidiate, X_test = train_test_split_no_unseen(X, test_size=test_size) 
        X_train, X_val = train_test_split_no_unseen(X_train_intermidiate, test_size=val_size) 
        
        X_train_reordered = X_train[:, [0, 2, 1]]  #swap column in ['from', 'to', 'rel'] format
        X_val_reordered = X_val[:, [0, 2, 1]]  #swap column in ['from', 'to', 'rel'] format
        X_test_reordered = X_test[:, [0, 2, 1]] #swap column in ['from', 'to', 'rel'] format
        X_full = np.concatenate([X_train_reordered,X_val_reordered,X_test_reordered])

        df_full = pd.DataFrame(X_full, columns=['from', 'to', 'rel'])
        df_train = pd.DataFrame(X_train_reordered, columns=['from', 'to', 'rel'])
        df_val = pd.DataFrame(X_val_reordered, columns=['from', 'to', 'rel'])
        df_test = pd.DataFrame(X_test_reordered, columns=['from', 'to', 'rel'])
        
        return df_full,df_train, df_val, df_test
    

def get_KnowledgeGraph(df_full ,df_train,df_test,df_val=None,):
    """
    Create KnowledgeGraph objects from DataFrame representations of triples.

    This function constructs a KnowledgeGraph for the full dataset and leverages its entity and relation mappings
    to generate corresponding KnowledgeGraph objects for the training, test, and optionally validation sets. All the input argoments must be 
    DataFrame representations in the [from, to, rel] format.

    Parameters
    ----------
    df_full : pandas.DataFrame
        DataFrame containing the full set of triples in the [from, to, rel] format .
    df_train : pandas.DataFrame
        DataFrame containing the training triples in the [from, to, rel] format .
    df_test : pandas.DataFrame
        DataFrame containing the test triples in the [from, to, rel] format .
    df_val : pandas.DataFrame, optional
        DataFrame containing the validation triples. If not provided, only the full, training, and test graphs are returned.

    Returns
    -------
 
    full_kg : KnowledgeGraph
            KnowledgeGraph representing the full dataset.
    train_kg : KnowledgeGraph
            KnowledgeGraph representing the training set.
    val_kg : KnowledgeGraph
            KnowledgeGraph representing the validation set.
    test_kg : KnowledgeGraph
            KnowledgeGraph representing the test set.

    """
    
    full_kg=KnowledgeGraph(df=df_full)
    train_kg = KnowledgeGraph(df=df_train,
        ent2ix=full_kg.ent2ix,  
        rel2ix=full_kg.rel2ix,  
        dict_of_heads=full_kg.dict_of_heads,  
        dict_of_tails=full_kg.dict_of_tails,  
        dict_of_rels=full_kg.dict_of_rels)
    
    test_kg = KnowledgeGraph(df=df_test,
        ent2ix=full_kg.ent2ix,  
        rel2ix=full_kg.rel2ix,  
        dict_of_heads=full_kg.dict_of_heads,  
        dict_of_tails=full_kg.dict_of_tails,  
        dict_of_rels=full_kg.dict_of_rels)
    
    if df_val is None:
        return full_kg,train_kg,test_kg
    else: 
        val_kg = KnowledgeGraph(df=df_val,
        ent2ix=full_kg.ent2ix,  
        rel2ix=full_kg.rel2ix,  
        dict_of_heads=full_kg.dict_of_heads,  
        dict_of_tails=full_kg.dict_of_tails,  
        dict_of_rels=full_kg.dict_of_rels)
        
        return full_kg,train_kg,val_kg,test_kg
    
    
def preprocess_unseen(directory_path,file_name,full_kg,add_reciproc=False):
    """
    Create a KnowledgeGraph for a set of new unseen triples using the original dataset mapping.

    This function loads unseen triples from a JSON file, reorders the columns to follow the [from, to, rel] format,
    and constructs a KnowledgeGraph for these triples. The entity and relation mappings from the original KnowledgeGraph
    (`full_kg`) are used to ensure consistency.

    Parameters
    ----------
    directory_path : str
        The folder where the JSON file is located.
    file_name : str
        The name of the JSON file.
    full_kg : KnowledgeGraph
        A KnowledgeGraph object representing the original dataset. Its entity and relation mappings (and related dictionaries)
        will be used to build the unseen KnowledgeGraph in the [from, to, rel] format .

    Returns
    -------
    X_unseen : ndarray of shape (n, 3)
        An array containing the new unseen triples in the format [from, to, rel].
    unseen_kg : KnowledgeGraph
        A KnowledgeGraph representing the set of new unseen triples.
    """
    X_unseen_from_json = load_pd_from_json(directory_path,file_name,add_reciproc).astype(str)
    X_unseen=X_unseen_from_json[:, [0, 2, 1]] #swap column in ['from', 'to', 'rel'] format
    
    unseen_df =  pd.DataFrame(X_unseen, columns=['from', 'to', 'rel'])
    
    unseen_kg = KnowledgeGraph(
    df=unseen_df,  # dataframe  (head, tail, rel)
    ent2ix=full_kg.ent2ix,  # map ent to indexes
    rel2ix=full_kg.rel2ix,  # map rel to indexes
    dict_of_heads=full_kg.dict_of_heads,  # dict valid heads
    dict_of_tails=full_kg.dict_of_tails,  # dict valid tails
    dict_of_rels=full_kg.dict_of_rels   # dict valid rel
)
    return X_unseen,unseen_kg


def check_for_unseen_file(DIR_PATH, file_name):
    
    unseen_file = DIR_PATH+file_name    
    
    
    if os.path.exists(unseen_file) and unseen_file.endswith('.json'):
        
        return True
    else:
        return False
        
    
    
def check_for_folder(DIR_PATH):    
    
    if os.path.exists(DIR_PATH) == False :
        
        os.mkdir(DIR_PATH)

        
