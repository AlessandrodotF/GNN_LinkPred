
import networkx as nx
import matplotlib.pyplot as plt
import textwrap
import random




def visualize_graph(global_dict,selected_category,selected_key):
    """
    Visualize a directed graph using NetworkX and Matplotlib.

    This function generates and displays a subgraph for a given key from the input graph dictionary.
    It formats node labels with line wrapping for better readability and visualizes the graph
    using a spring layout.

    Args:
        graph (dict): A dictionary where each key represents a node, and the associated value 
                      is a list of dictionaries containing:
                      - "dest" (str): The destination node UUID.
                      - "taxonomy_url" (str or list): The taxonomy information (either a string or a list of strings).
        key (str): The unique identifier of the source node to visualize.


    Visualization:
        - Nodes are drawn in light blue.
        - Edges are directed (arrows).
        - Node labels are wrapped for better readability.
        - Edge labels correspond to the 'taxonomy_url' attribute.
        - The graph layout is computed using the spring layout algorithm.

    Returns:
        None: The function directly visualizes the graph using Matplotlib.
    """
    wrap_length=15
    
    # init empty directed graph
    G = nx.DiGraph()
    G.add_node(selected_key)

    node_set = global_dict[selected_category].get(selected_key, [])
   
        
    # dict of labels for nodes with formatted text
    node_labels = {selected_key: "\n".join(textwrap.wrap(selected_key, wrap_length))}
    
    for node_info in node_set:
        
        dest = node_info["dest"]
        # Se taxonomy_url è una lista, la converte in stringa
        label = ",".join(node_info["taxonomy_url"]) if isinstance(node_info["taxonomy_url"], list) else node_info["taxonomy_url"]
        G.add_node(dest)
        G.add_edge(selected_key, dest, label=label)
        node_labels[dest] = "\n".join(textwrap.wrap(dest, wrap_length))

            
    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'label')
    plt.figure(figsize=(20, 8))  
    nx.draw(G, pos, labels=node_labels, with_labels=True, node_color='lightblue', 
            node_size=1500, arrowstyle='->', arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(f"G: {selected_key}")
    plt.show()
    
    
def visualize_n_step(G1, label_string="all"):
    """
    Visualize a directed or undirected graph using NetworkX and Matplotlib.

    Args:
        G1 (networkx.Graph or networkx.DiGraph): The graph to visualize (directed or undirected).
        label_string (str, optional): Specifies the type of labels for the nodes.
            - "uuid": Displays only the UUID of each node.
            - "taxonomy_url": Displays only the taxonomy URL of each node.
            - "all" (default): Displays both UUID and taxonomy URL.
    """
    # label on nodes
    if label_string == "uuid":
        labels = {node: G1.nodes[node]['uuid'] for node in G1.nodes()}
    elif label_string == "taxonomy_url":
        labels = {node: G1.nodes[node]['taxonomy_url'] for node in G1.nodes()}
    else:
        labels = {node: f"uuid: {G1.nodes[node]['uuid']}\nTaxonomy: {G1.nodes[node]['taxonomy_url']}" for node in G1.nodes()}
    # Color nodes, red=source otherwise blue
    node_colors = ['red' if G1.nodes[node].get('source') is not None else 'lightblue' for node in G1.nodes()]
    
    # layout
    pos = nx.circular_layout(G1)
    plt.figure(figsize=(20, 8))
    if G1.is_directed():
        nx.draw(G1, pos, node_color=node_colors, node_size=1500, arrowstyle='->', arrowsize=20)
    else:
        nx.draw(G1, pos, node_color=node_colors, node_size=1500)
    # label for edges
    edge_labels = nx.get_edge_attributes(G1, 'label')
    nx.draw_networkx_edge_labels(G1, pos, edge_labels=edge_labels, font_size=8)
    # label for nodes
    nx.draw_networkx_labels(G1, pos, labels=labels, font_size=8, font_weight='bold')
    plt.title(f"G1")
    plt.show()



def build_graph_networkx(graph, key):
    """
    Build a directed graph using NetworkX from a given atomic graph structure. the Directed attribute is just a trick to doble check the results on small graphs

    This function initializes a directed graph (`DiGraph`) and constructs edges 
    starting from a specified key (source node). It retrieves the corresponding 
    subgraph from the input dictionary and adds edges to the destination nodes 
    with labels based on the `taxonomy_url` field.

    Args:
        graph (dict): A dictionary representing the atomic graph structure, 
                      where keys are source UUIDs and values are lists of dictionaries
                      with destination UUIDs and associated taxonomy URLs.
        key (str): The UUID of the source node for which the directed subgraph will be built.

    Returns:
        networkx.DiGraph: A directed graph (`DiGraph`) where nodes represent UUIDs 
                          and edges are labeled with `taxonomy_url`.
    """

    # init empty directed graph
    G = nx.DiGraph()

    # add starting node (key variable)
    G.add_node(key)
    # select sub_graph for the specified key
    node_set = graph.get(key, [])
    for node_info in node_set:
        dest = node_info["dest"]
        # if taxonomy_url is a list, convert in string
        label = ",".join(node_info["taxonomy_url"]) if isinstance(node_info["taxonomy_url"], list) else node_info["taxonomy_url"]
        G.add_node(dest)
        G.add_edge(key, dest, label=label)
    return G


def build_atomic_graphs(rel_list, obj_list):
    """
    Build an atomic graph where each source node corresponds to the UUID found in the src key of a relation entry. 
    Then, retrieve all destination UUIDs (dest keys) from the relations dataset and link them to their respective source node.
    Each atomic graph will be uniquely identified by the UUID of the source relation.

    Args:
        rel_list (list[dict]): A list of dictionaries where each entry contains (src, dest, taxonomy_url) keys.
        obj_list (list[dict]): A list of dictionaries where each entry contains (label, uuid, taxonomy_url) keys.

    Returns:
        dict: A dictionary of atomic graphs, where each key is a unique UUID identifier.
        list: A list of the keys of the output dictionary.
    """

    # dict of (uuid, objs) pair key value 
    obj_dict = {obj["uuid"]: obj for obj in obj_list} #ok
    # init graph: every uuid = starting node
    graph_src = {obj : [] for obj in obj_dict}
    graph_dest = {obj : [] for obj in obj_dict}

    # For every relarion entry, check if  (src) and (dest) exist in obj_dict 
    for rel in rel_list:
        src = rel["src"]
        dest = rel["dest"]
        if src in obj_dict and src != dest: #esplicito anche il caos di self loop 
            # SAVE FIRST STEP GRAPH e anche il "receiver" volendo
            #graph[src].append({"dest": dest, "taxonomy_url": rel["taxonomy_url"], "rec" : obj_dict[dest]})
            graph_src[src].append({"dest": dest, "taxonomy_url": rel["taxonomy_url"]})
            
        if  dest in obj_dict and src != dest: #esplicito anche il caos di self loop 
            # SAVE FIRST STEP GRAPH e anche il "receiver" volendo
            #graph[src].append({"dest": dest, "taxonomy_url": rel["taxonomy_url"], "rec" : obj_dict[dest]})
            graph_dest[dest].append({"dest": src, "taxonomy_url": rel["taxonomy_url"]})

    graphs_src = {k: v for k, v in graph_src.items() if v}
    graphs_dest = {k: v for k, v in graph_dest.items() if v}
    
    global_dict={"src_dict":graphs_src, "dest_dict":graphs_dest}
    global_valid_keys={"src_dict": list(graphs_src.keys()),"dest_dict": list(graphs_dest.keys())}

    return global_dict,global_valid_keys
    




def select_order(similarity_metric_src, similarity_metric_dest):
    """Select the correct order for the merging procedure between "src_dict" and "dest_dict".
    Args:
        similarity_metric_src (list): Contains similarity measure (overlap of nodes) and the total number of nodes.
        similarity_metric_dest (list): Contains similarity measure (overlap of nodes) and the total number of nodes.
    Returns:
        tuple: (first_category, second_category)
    """
    (sim_src, size_src), (sim_dest, size_dest) = similarity_metric_src, similarity_metric_dest

    if (sim_src, size_src) > (sim_dest, size_dest):  # direct comparison of tuple
        return "src_dict", "dest_dict"
    if (sim_src, size_src) < (sim_dest, size_dest):
        return "dest_dict", "src_dict"
    
    return random.choice([("src_dict", "dest_dict"), ("dest_dict", "src_dict")])


def compute_similarity(graph_dict, main_graph):
    """
    Computes a similarity measure between the main graph and a set of graphs contained in a reference dictionary.
    The similarity is based on the number of nodes that are common between the main graph and each graph in the dictionary.
    Args:
        graph_dict (dict): A dictionary where the keys are identifiers (or graph names) and the values are NetworkX graph objects. 
                           Each graph represents a subgraph that will be compared with the main graph.
        main_graph (networkx.Graph): The graph that needs to be expanded, representing the main graph to which the graphs in `graph_dict` will be compared.
    Returns:
        tuple: A tuple containing two values:
            - `nodes_in_common` (int): The total number of nodes common between the `main_graph` and all the graphs in `graph_dict`.
            - `total_nodes` (int): The total number of nodes across all graphs in `graph_dict`.

    """
    #extende version of what is written below
    #list_of_graph = graph_dict.values()
    #global_counter=[]
    #for element in list_of_graph: ##seleziona il singolo grafo da graph_dict
    #    conter_for_element=[]
    #    for node in element.nodes: ## itera sui nodi del grafo selezionato
    #        if node in main_graph.nodes: 
    #            conter_for_element.append(1) ## +1 se il nodo è presente in main_graph
    #    global_counter.append(sum(conter_for_element))
    #similarity = sum(global_counter)  
    ## dato main graph , la misura di similarita mi dice a quale gruppo è piu simile tra i due 
    ## il criterio è il numero di nodi in comune
    nodes_in_common = sum(
        sum(node in main_graph.nodes for node in G.nodes)  # node in common
        for G in graph_dict.values()
        )
    total_nodes = sum(len(G.nodes) for G in graph_dict.values())  # total number of nodes
    return [nodes_in_common, total_nodes]

def build_graph_n_steps(dict_graph_src,dict_graph_dest, selected_category, selected_key, uuid_taxonomy_obj_dict):

    """
    Builds an expanded directed graph starting from a smaller graph identified by selected_category (src_dict or dest_dict) 
    and selected_key (UUID).
    The process consists of:
    1. Identifying the most similar set of graphs (from `src_dict` or `dest_dict` inside dict_graphs) to the initial small graph based on node overlap.
    2. Sorting the identified groups of graphs by similarity.
    3. Merging the sorted graphs into the main graph in decreasing order of similarity.
    4. Assigning original node attributes.

    Args:
        dict_graph_src (dict): Dictonary pof atomic graphs starting from the src node as starting point
        dict_graph_dest (dict): Dictonary pof atomic graphs starting from the dest node as starting point
        selected_category (str): Key to select the sub-dictionary ("src_dict" or "dest_dict").
        selected_key (str): Identifier of the specific graph inside a selected category.
        uuid_taxonomy_obj_dict (dict): Dictionary mapping UUIDs to taxonomy URLs.

    Returns:
        networkx.DiGraph: A directed graph (`DiGraph`) obtained by merging similar graphs while preserving node attributes.
    """


    # union
    dict_graphs = {"src_dict": dict_graph_src, "dest_dict": dict_graph_dest}

    # select main graph and delete it
    main_graph = dict_graphs[selected_category][selected_key]
    del dict_graphs[selected_category][selected_key]
 

    #Select the correct sub group of graphs from which the merging will start
    similarity_metric_src = compute_similarity(dict_graphs["src_dict"], main_graph)
    similarity_metric_dest = compute_similarity(dict_graphs["dest_dict"], main_graph)
    first_category,second_category = select_order(similarity_metric_src,similarity_metric_dest)


    
    def merge_graphs_from_category(category):
        """
        Merges graphs from a specified category based on their similarity to the main graph.
        The similarity is determined by the number of nodes that are common between each graph in the category and the main graph.
        The graphs are merged in decreasing order of similarity, and only graphs that have at least one common node with the main graph are included.

        Args:
            category (str): The category key (src_dict or dest_dict) used to fetch the graphs from the `dict_graphs`. The graphs under this category will be merged.

        Returns:
            None: The function modifies the `main_graph` in place by adding nodes and edges from the graphs in the specified category.
        """
        graph_list = [
            (key, G2, sum(node in main_graph.nodes for node in G2.nodes))  # Conta nodi in comune
            for key, G2 in dict_graphs[category].items()
        ]

        # order graph for similarity (decreasing way)
        #ricorda
        #lanbda = nome funzione, x è l argomento, x[2] il suo return
        graph_list.sort(key=lambda x: x[2], reverse=True)  
  
        # merge
        for _, G2, _ in graph_list:
            # does G2 have at least 1 node in common? add all
            if any(node in main_graph.nodes() for node in G2.nodes()):
                main_graph.add_nodes_from(G2.nodes)
                for u, v in G2.edges():
                    edge_data = G2.get_edge_data(u, v)
                    label = edge_data.get('label', "N/A")
                    main_graph.add_edge(u, v, label=label)

    # the most similar first
    merge_graphs_from_category(first_category)
    merge_graphs_from_category(second_category)

    # encode attributes 
    for node in main_graph.nodes():
        taxonomy = uuid_taxonomy_obj_dict.get(node, "N/A")
        main_graph.nodes[node]['uuid'] = node
        main_graph.nodes[node]['taxonomy_url'] = taxonomy
        

    return main_graph



def compress_graph_nx(G, interesting_labels):
    """
    Compresses the original graph by keeping only nodes with an interesting `taxonomy_url` 
    and directly reconnecting them, removing intermediate non-relevant nodes.

    Args:
        G (networkx.Graph): The input graph, represented as a NetworkX DiGraph or Graph.
        interesting_labels (list of str): A list of categories of interest (e.g., ["#Person", "#Organisation"]).
            Only nodes with a `taxonomy_url` matching one of these labels will be retained.

    Returns:
        networkx.Graph: A new undirected graph that retains only the nodes of interest 
                         and directly reconnects those that were previously linked through intermediate nodes.
    """
    
    G = G.to_undirected()
    new_graph = nx.Graph()

    # Collect interesting nodes and their neighbors
    interesting_nodes = [node for node in G.nodes if G.nodes[node].get("taxonomy_url") in interesting_labels]
    interesting_neighbors = {}

    
    for node in interesting_nodes:
        new_graph.add_node(node, **G.nodes[node])
        
        # Check the neighbors of the interesting node
        for neighbor in G.neighbors(node):

            if G.nodes[neighbor].get("taxonomy_url") in interesting_labels:
                # Add edges between interesting nodes only
                new_graph.add_edge(node, neighbor, label="#IsRelated")
            else:
                # Store non-interesting neighbors to handle later
                if neighbor not in interesting_neighbors:
                    interesting_neighbors[neighbor] = []
                interesting_neighbors[neighbor].append(node)

    # Now handle non-interesting nodes and connect their interesting neighbors
    for node, neighbors in interesting_neighbors.items():
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                new_graph.add_edge(neighbors[i], neighbors[j], label="#IsRelated")

    return new_graph


