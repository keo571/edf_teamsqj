"""
This module provides functions for pre-processing JSON data from the `raw_data` folder.
"""

import os
import re
import json
import graph
import pickle
import pandas as pd
from typing import Tuple, List


###### First Iteration ######

def first_iter_pipeline(politician:graph.Node, vote:json, donors:json, match:json, min_amount=2000, min_rating=60) -> Tuple[List, List]:
    dump({'politician_nodes':[politician]}, iter_name='first_iter', state=politician.state)
    # Create policy nodes and vote edges between a politician and the policies they vote for
    vote_dict = policy_nodes_and_vote_edges(politician.name, vote)
    dump(vote_dict, iter_name='first_iter', state=politician.state)
    # Create donor nodes and doante edges between the politician and ther donors
    donor_dict = donor_nodes_and_donate_edges(politician.name, donors, min_amount=min_amount)
    dump(donor_dict, iter_name='first_iter', state=politician.state)
    # Create interest group nodes and rate edges between the politician and the interest groups that rate them
    interest_group_dict = interest_group_nodes_and_rate_edges(politician.name, match, min_rating=min_rating)
    dump(interest_group_dict, iter_name='first_iter', state=politician.state)
    return donor_dict['donate_edges'], interest_group_dict['rate_edges']


def policy_nodes_and_vote_edges(politician_name:str, vote:json) -> dict:
    policy_nodes = []
    vote_edges = []
    for bill in vote[politician_name]:
        # Create policy nodes
        policy = graph.Policy(name=bill['Bill No.'], date=bill['Date'], synopsis=bill['Synopsis'], 
                                  outcome=bill['Outcome'].replace('  ', ' '))
        policy_nodes.append(policy)  
        # Create vote edges
        vote_edge = graph.Vote(source_node=politician_name, target_node=bill['Bill No.'])
        vote_edges.append(vote_edge)

    vote_dict = {'policy_nodes': policy_nodes, 'vote_edges':vote_edges}
    return vote_dict


def donor_nodes_and_donate_edges(politician_name:str, donors:json, min_amount:int) -> dict:
    donor_nodes = []
    donate_edges = []
    name = politician_name.split("(")[0].strip()
    for donor in donors[politician_name]:
        # Convert the amount to a float
        amount = float(re.sub(r'[^\d.]', '', donor['amount'])) 
        if amount >= min_amount: 
            # Create donor nodes
            if donor['name'] == 'UNITEMIZED DONATIONS':
                donor_node = graph.Donor(name=f'{donor["name"]} - {name}')
            else:
                donor_node = graph.Donor(name=donor['name'])
            donor_nodes.append(donor_node)
            # Create donate edges
            donate_edge = graph.Donate(source_node=donor_node.name, target_node=name, weight=amount)
            donate_edges.append(donate_edge)
    
    donor_dict = {'donor_nodes': donor_nodes, 'donate_edges':donate_edges}
    return donor_dict


def interest_group_nodes_and_rate_edges(politician_name:str, match:json, min_rating:int) -> dict:
    interest_group_nodes = []
    rate_edges = []
    pattern = re.compile(r'^([^\(\)]+)')
    for interest_group in match[politician_name]:
        # Ignore the interest group that gave letter grade
        try:
            rating = float(re.sub(r'[^\d.]', '', interest_group['match']))
        except ValueError:
            rating = 0.0

        if rating >= min_rating:
            # Create interest group nodes
            group_node = graph.InterestGroup(name=interest_group['name'])
            interest_group_nodes.append(group_node)
            # Create rate edges
            match = pattern.search(politician_name)
            politician_name = match.group(1).strip()
            rate_edge = graph.Rate(source_node=group_node.name, target_node=politician_name, weight=rating)
            rate_edges.append(rate_edge)
    
    interest_group_dict = {'interest_group_nodes': interest_group_nodes, 'rate_edges':rate_edges}
    return interest_group_dict


def dump(file_dict:dict, iter_name:str, state:str) -> None:
    path = os.path.join('processed_data', f'{state}', f'{iter_name}')
    os.makedirs(path, exist_ok=True)
    for file_name in file_dict.keys():
        with open(f'{path}\{file_name}.pkl', 'wb') as f:
            pickle.dump(file_dict[file_name], f)


###### Second Iteration ######

def second_iter_pipeline(politician:graph.Node, vote:json) -> List[graph.Node]:
    second_depth_dict = second_depth_politician_nodes_and_edges(politician, vote)
    dump(second_depth_dict, iter_name='second_iter', state=politician.state)
    return second_depth_dict['second_depth_politician_nodes']


def second_depth_politician_nodes_and_edges(first_depth_politician:graph.Node, vote:json) -> dict:
    pattern1 = r'(.+) \((\w+) - (\w+)\)'
    pattern2 = r'^(.*) \(.+\)$'
    politician_nodes = []
    committee_nodes = []
    sponsor_edges = []
    consensus_edges = []

    for bill in vote[first_depth_politician.name]:
        policy_name = bill['Bill No.'] 
        politician_nodes_bill = []
        
        # Create second depth politician nodes 
        for sponsor in bill['Sponsor'] + bill['Cosponsor']:
            result = re.match(pattern1, sponsor).groups()
            politician_node = graph.Politician(name=result[0], party=result[2], state=result[1])
            politician_nodes_bill.append(politician_node)
            # Create sponsor edges between second depth politicians and the bills they sponsor
            sponsor_edge = graph.Sponsor(source_node=result[0], target_node=policy_name)
            sponsor_edges.append(sponsor_edge)
        
        # Create consensus edges between all second depth politicians who sponsor the same bill
        for i in range(len(politician_nodes_bill)):
            for j in range(i+1, len(politician_nodes_bill)):
                # if the politicians are in the same party, use default weight 1
                if politician_nodes_bill[i].party ==  politician_nodes_bill[j].party:
                    consensus_edges.append(graph.Consensus(source_node=politician_nodes_bill[i].name, target_node=politician_nodes_bill[j].name))
                    consensus_edges.append(graph.Consensus(source_node=politician_nodes_bill[j].name, target_node=politician_nodes_bill[i].name))
                # else set weight to 0.5
                else:
                    consensus_edges.append(graph.Consensus(source_node=politician_nodes_bill[i].name, target_node=politician_nodes_bill[j].name, weight=0.5))
                    consensus_edges.append(graph.Consensus(source_node=politician_nodes_bill[j].name, target_node=politician_nodes_bill[i].name, weight=0.5))
        politician_nodes.extend(politician_nodes_bill)
        
        # Create committee nodes
        for sponsor in bill['Committee']:
            result = re.match(pattern2, sponsor).group(1)
            committee_node = graph.Committee(name=result, state=first_depth_politician.state)
            committee_nodes.append(committee_node)
            # Create sponsor edges for committee nodes
            sponsor_edge = graph.Sponsor(source_node=result, target_node=policy_name)
            sponsor_edges.append(sponsor_edge) 

    # Create consensus edges between all second depth politicians and that first depth politician
    for i in range(len(politician_nodes)):
        consensus_edges.append(graph.Consensus(source_node=politician_nodes[i].name, target_node=first_depth_politician.name))
        consensus_edges.append(graph.Consensus(source_node=first_depth_politician.name, target_node=politician_nodes[i].name))
    
    second_depth_dict = {'second_depth_politician_nodes': politician_nodes, 'committee_nodes':committee_nodes, 'sponsor_edges':sponsor_edges,
                         'consensus_edges':consensus_edges}
    
    return  second_depth_dict


###### Third Iteration ######

# See data_processing.py


###### Fourth Iteration ######

# See data_processing.py

###### Extra Step for Geospatial Intelligence ######

def geo_int(edges:List[graph.Edge], first_depth_politicians:List[graph.Node], type:str) -> None:
    """
    Argument:
    edges -- a list of edges
    first_depth_politicians -- a list of first depth politician nodes
    type -- 'donor' or 'interest_group'
    """
    # Create a DataFrame from an edge list
    df= pd.DataFrame([vars(edge) for edge in edges])
    # Filter the DataFrame to exclude the individuals whose addresses cannot be traced
    mask = ~df['source_node'].str.contains(',')
    filtered_df = df[mask]
    # Create a DataFrame for the first depth politicians
    politician_df= pd.DataFrame([vars(node) for node in first_depth_politicians])
    # Merge the two DataFrames
    merged_df = pd.merge(filtered_df, politician_df, left_on='target_node', right_on='name')
    merged_df = merged_df.drop(['name', 'node_type'], axis=1).rename(columns={'source_node': f'{type}', 'target_node': 'politician'})  
    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(f'processed_data/geoint_{type}.csv', index=False)

