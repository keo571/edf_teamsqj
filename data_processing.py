import json
import graph
import pandas as pd
from typing import List
import pre_processing as pre


###### Step 0: Read raw data files ######

directory = 'raw_data/'
file_names = ['votesmart_vote.json', 'votesmart_donors.json', 'votesmart_donors_second_depth.json', 'votesmart_match.json', 
              'votesmart_match_second_depth.json']

data_list = []
for file_name in file_names:
    with open(directory+file_name, 'r') as f:
        data = json.load(f)
        data_list.append(data)


###### First Iteration ######

politician1 = graph.Politician(name='Karen Spilka', party='D', state='MA')
politician2 = graph.Politician(name='Liz Krueger', party='D', state='NY')
politician3 = graph.Politician(name='Ana Rodriguez', party='R', state='FL')
first_depth_politicians = [politician1, politician2, politician3]
donate_edge_list = []
rate_edge_list = []

for politician in first_depth_politicians:
    donate_edges, rate_edges = pre.first_iter_pipeline(politician, data_list[0], data_list[1], data_list[3], min_amount=2000, min_rating=60)
    donate_edge_list.extend(donate_edges)
    rate_edge_list.extend(rate_edges)


###### Second Iteration ######

second_depth_politician_dict = {}

for politician in first_depth_politicians:
    second_depth_politicians = pre.second_iter_pipeline(politician, data_list[0])
    second_depth_politician_dict[politician.state] = second_depth_politicians


###### Third Iteration ######

for state, politicians in second_depth_politician_dict.items():
    second_depth_donor_nodes = []
    second_depth_donate_edges = []
    for politician in politicians:
        revised_name = f'{politician.name} ({state} - {politician.party})'
        donor_dict = pre.donor_nodes_and_donate_edges(revised_name, data_list[2], min_amount=2000)
        second_depth_donor_nodes.extend(donor_dict['donor_nodes'])
        second_depth_donate_edges.extend(donor_dict['donate_edges'])

    second_depth_donor_dict = {'second_depth_donor_nodes': second_depth_donor_nodes, 'second_depth_donor_edges': second_depth_donate_edges}
    pre.dump(second_depth_donor_dict, iter_name='third_iter', state=state)


###### Fourth Iteration ######

for state, politicians in second_depth_politician_dict.items():
    second_depth_interest_group_nodes = []
    second_depth_rate_edges = []
    for politician in politicians:
        revised_name = f'{politician.name} ({state} - {politician.party})'
        interest_group_dict = pre.interest_group_nodes_and_rate_edges(revised_name, data_list[4], min_rating=60)
        second_depth_interest_group_nodes.extend(interest_group_dict['interest_group_nodes'])
        second_depth_rate_edges.extend(interest_group_dict['rate_edges'])

    second_depth_interest_group_dict = {'second_depth_interest_group_nodes': second_depth_interest_group_nodes, 'second_depth_rate_edges': second_depth_rate_edges}
    pre.dump(second_depth_interest_group_dict, iter_name='fourth_iter', state=state)


###### Extra Step for Geospatial Intelligence ######

pre.geo_int(donate_edge_list, first_depth_politicians, type='donor')
pre.geo_int(rate_edge_list, first_depth_politicians, type='interest_group')
