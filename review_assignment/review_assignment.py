# This script solves the review assignment problem for a HotCRP instance. 
# The first version was developed by Madan Musuvathi and Dan Tsafrir for
# use during the three reviewing cycles of ASPLOS '24 
# The current version is a simplified version, currently being used for
# PPoPP '26 by Madan Musuvathi and Kenjiro Taura

from argparse import ArgumentParser
from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import sys
import os

arg_parser = ArgumentParser()
arg_parser.add_argument('-i', '--hotcrp_instance', type=str, help='HotCRP instance name', default='ppopp26')
arg_parser.add_argument('-r', '--round', type=str, help='round for the review assignment: R1, R2', default = 'R1')
arg_parser.add_argument('-o', '--output_prefix', type=str, help='output prefix')
args = arg_parser.parse_args()

hotcrp_instance = args.hotcrp_instance
round = args.round

# input files, should be present in the current directory
pc_info_file = hotcrp_instance+'-pcinfo.csv'    # https://<hotcrp_instance>.hotcrp.com/users?t=pc, select all, download "PC Info"
papers_file = hotcrp_instance+'-data.csv'   # https://<hotcrp_instance>.hotcrp.com/search?q=&t=s, select all, download “Paper Information -> CSV". 
allprefs_file = hotcrp_instance+'-allprefs.csv' # https://<hotcrp_instance>.hotcrp.com/search?q=&t=s, select all, download “PC Review Preferences”. 
tpms_score_file = hotcrp_instance+'-tpms-scores.csv'  # a headerless csv file <paper>,<tpms_email>,<tpms_score> as provided by Laurent
tpms_alias_file = hotcrp_instance+'-tpms-aliases.csv'  # a csv file <tpms_email>,<email> as provided to Laurent
pc_load_file = hotcrp_instance+'-pcload.csv'  # <email>,<max_load> 

# round specific args
if round == 'R1':
    min_reviews_per_paper = 3
    pre_assigned_reviews_df = None
elif round == 'R2':
    min_reviews_per_paper = 5
    pre_assigned_reviews_df = pd.read_csv(f'{hotcrp_instance}-R1-assignments.csv')
else:
    raise ValueError(f"Unknown round: {round}")

# these are output files
output_prefix = args.output_prefix if args.output_prefix else hotcrp_instance
assignment_file = f'{output_prefix}-{round}-assignments.csv' # can be used directly in HotCRP
assignment_details_file = f'{output_prefix}-{round}-assignments-details.csv'

def main():
    pcinfo_df = load_pcs(pc_info_file, pc_load_file)
    papers_df = load_papers(papers_file, min_reviews_per_paper)
    topics_df = load_topic_scores(allprefs_file, pcinfo_df, papers_df)
    tpms_df = load_tpms_scores(tpms_score_file, tpms_alias_file, pcinfo_df, papers_df)

    assignments_df = solve_review_assignments(pcinfo_df, papers_df, topics_df, tpms_df, pre_assigned_reviews_df)
    save_assignments(assignments_df, assignment_file)
    analyze_assignments(assignments_df, pcinfo_df, papers_df, topics_df, tpms_df, pre_assigned_reviews_df)

def print_error(msg):
    print(f"\033[91m{msg}\033[0m")

def load_pcs(pc_info_file, pc_load_file):
    pcinfo_df = pd.read_csv(pc_info_file)
    # remove chairs from pcinfo_df
    pcinfo_df = pcinfo_df[~pcinfo_df['roles'].str.contains('chair', na=False)]

    pcinfo_df.fillna(0, inplace=True)
    process_topic_columns(pcinfo_df)

    load_df = pd.read_csv(pc_load_file)
    pcinfo_df = pcinfo_df.merge(load_df, on="email", how="left")
    missing_load = pcinfo_df[pcinfo_df["max_load"].isna()]["email"].tolist()
    if missing_load:
        print_error(f"PC members without load info in {pc_load_file}: {missing_load}")

    print(f"Loaded {pcinfo_df.shape[0]} PC members from {pc_info_file}")
    return pcinfo_df

def process_topic_columns(pcinfo_df):
    topic_columns = pcinfo_df.columns[pcinfo_df.columns.str.startswith("topic:")]
    # find PC members who have not updated their topics: all topic_columns are zero
    no_topic_entries = pcinfo_df[(pcinfo_df[topic_columns] == 0).all(axis=1)]
    if not no_topic_entries.empty:
        print_error(f"PC members who have not updated their topics: {no_topic_entries['email'].tolist()}")

    # some PC members do not use negative preferences, or use -1 instead of -2
    # so make their min value as -2
    row_mins = pcinfo_df[topic_columns].min(axis=1)
    mask = pcinfo_df[topic_columns].eq(row_mins, axis=0)
    pcinfo_df.loc[:, topic_columns] = pcinfo_df[topic_columns].mask(mask, -2)

def load_papers(papers_file, min_reviews_per_paper):
    papers_df = pd.read_csv(papers_file)
    papers_df = papers_df.rename(columns={'ID':'paper'})
    papers_df.fillna(0, inplace=True)
    papers_df['min_reviews'] = min_reviews_per_paper
    print(f"Loaded {papers_df.shape[0]} papers from {papers_file}")
    return papers_df

def load_topic_scores(allprefs_file, pcinfo_df, papers_df):
    topic_df = pd.read_csv(allprefs_file)
    topic_df.drop(columns=['title', 'given_name', 'family_name'], inplace=True)
    # drop chairs by removing emails that are not in pcinfo_df
    topic_df = topic_df[topic_df['email'].isin(pcinfo_df['email'])]
    topic_df.fillna(0, inplace=True)

    # normalize topic_scores
    topic_df = normalize_scores(topic_df, 'topic_score')

    # find PC members who have not updated their preferences, all their preferences are zero
    all_zero_pref = topic_df.groupby("email")["preference"].apply(lambda x: (x == 0).all())
    if all_zero_pref.any():
        print_error(f"PC members who have not updated their preferences: {all_zero_pref[all_zero_pref].index.tolist()}")

    # these values normalize topic_scores to -1 to 1, just as tpms_scores 
    expert_score = 1
    like_score = 0.5
    dont_want_score = -1
    conflict_score = -999

    topic_df["preference"] = topic_df["preference"].apply(
                                        lambda x: expert_score       if x >= 20       # likely to provide expert review
                                                else like_score      if 0 < x < 20    # love to  review
                                                else dont_want_score if -999 < x <= 0 # dont want to review
                                                else conflict_score                   # conflicted
                                        )

    # chairs should carefully look at conflicts stated by PC members
    conflicting_pref = topic_df[topic_df["preference"] == conflict_score]
    if not conflicting_pref.empty:
        # print email and paper
        for _, row in conflicting_pref.iterrows():
            print_error(f"PC member {row['email']} has conflicting preference for paper {row['paper']}")

    # mark 'conflict' if preference == conflict_score
    topic_df.loc[topic_df["preference"] == conflict_score, 'conflict'] = 'conflict'
    topic_df["preference"] = topic_df["preference"].replace(conflict_score, 0)

    # find PC members who didnt mark sufficient number of expert reviews
    expert_reviews = topic_df[topic_df["preference"] == expert_score]
    expert_counts = expert_reviews.groupby("email").size().reset_index(name="expert_count")
    merged = expert_counts.merge(pcinfo_df[["email", "max_load"]], on="email", how="left")
    underloaded = merged[merged["expert_count"] < 1.5 * merged["max_load"]]
    if not underloaded.empty:
        print_error(f"PC members who didn't mark sufficient expert reviews: {underloaded['email'].tolist()}")

    # find PC members who didnt mark sufficient number of expert/like reviews
    like_or_expert_reviews = topic_df[topic_df["preference"] >= like_score]
    like_or_expert_counts = like_or_expert_reviews.groupby("email").size().reset_index(name="like_or_expert_count")
    merged = like_or_expert_counts.merge(pcinfo_df[["email", "max_load"]], on="email", how="left")
    underloaded = merged[merged["like_or_expert_count"] < 2 * merged["max_load"]]
    if not underloaded.empty:
        print_error(f"PC members who didn't mark sufficient expert/like reviews: {underloaded['email'].tolist()}")

    # Find papers that dont have sufficient expert reviews
    expert_counts = expert_reviews.groupby("paper").size().reset_index(name="expert_count")
    merged = expert_counts.merge(papers_df[["paper", "min_reviews"]], on="paper", how="left")
    underloaded = merged[merged["expert_count"] < 1.5 * merged["min_reviews"]]
    if not underloaded.empty:
        print_error(f"Papers with insufficient expert reviews: {underloaded['paper'].tolist()}")

    # Find papers that dont have sufficient expert/like reviewers
    like_or_expert_counts = like_or_expert_reviews.groupby("paper").size().reset_index(name="like_or_expert_count")
    merged = like_or_expert_counts.merge(papers_df[["paper", "min_reviews"]], on="paper", how="left")
    underloaded = merged[merged["like_or_expert_count"] < 2 * merged["min_reviews"]]
    if not underloaded.empty:
        print_error(f"Papers with insufficient expert/like reviewers: {underloaded['paper'].tolist()}")

    print(f"Loaded {topic_df.shape[0]} scores and preferences from {allprefs_file}")
    return topic_df

def normalize_scores(df, score):
    df_mean = df[score].mean()
    df_max = df[score].max()
    df_min = df[score].min()
    new_col = 'norm_' + score
    #use min-max normalization
    df[new_col] = df[score].apply(lambda x: 2 * (x - df_min) / (df_max - df_min) - 1)
    return df

def load_tpms_scores(tpms_score_file, tpms_alias_file, pcinfo_df, papers_df):
    tpms_df = pd.read_csv(tpms_score_file, header=None, names=['paper','tpms_email','tpms_score'])
    tpms_alias = pd.read_csv(tpms_alias_file)
    # merge and drop tpms_email
    tpms_df = tpms_df.merge(tpms_alias, on='tpms_email', how='left')
    tpms_df.drop(columns=['tpms_email'], inplace=True)

    # find PC members that are missing tpms_scores
    missing_tpms = pcinfo_df[~pcinfo_df['email'].isin(tpms_df['email'])]
    if not missing_tpms.empty:
        print_error(f"PC members missing tpms_scores: {missing_tpms['email'].tolist()}")

    # find papers that are missing tpms_scores
    missing_papers = papers_df[~papers_df['paper'].isin(tpms_df['paper'])]
    if not missing_papers.empty:
        print_error(f"Papers missing tpms_scores: {missing_papers['paper'].tolist()}")

    print(f"Loaded {tpms_df.shape[0]} tpms scores from {tpms_score_file}")
    return tpms_df

def solve_review_assignments(pcinfo_df, papers_df, topics_df, tpms_df, pre_assigned_reviews_df=None):
    pc_emails = pcinfo_df['email'].tolist()
    paper_ids = papers_df['paper'].tolist()

    # do some sanity checks
    max_reviews_available = sum(pcinfo_df['max_load'])
    min_reviews_required = sum(papers_df['min_reviews'])

    if max_reviews_available < min_reviews_required:
        print_error(f"Infeasible problem: max reviews available {max_reviews_available} is less than min reviews required{min_reviews_required}.")
        return None

    review_slack = max_reviews_available - min_reviews_required
    # uniformly spread the slack across all PC members (may be, differentiate between full and erc)
    slack_per_pc = math.ceil(review_slack / pcinfo_df.shape[0])

    model = cp_model.CpModel()
    log_file = f'{hotcrp_instance}-{round}-cpsat.log'
    logf = open(log_file, 'w')

    assign = {}
    # for each pc and each paper, create an assign_{email, paper} variable
    for email in pc_emails:
        for paper in paper_ids:
            assign[email, paper] = model.NewBoolVar(f'assign[{email}, {paper}]')


    # constraint min_reviews per paper
    for index, row in papers_df.iterrows():
        paper = row['paper']
        min_reviews = row['min_reviews']
        model.Add(sum(assign[email, paper] for email in pc_emails) == int(min_reviews))

    # constraint load per pc
    for index, row in pcinfo_df.iterrows():
        email = row['email']
        max_load = int(row['max_load'])
        min_load = max_load - slack_per_pc
        model.Add(sum(assign[email, paper] for paper in paper_ids) <= max_load)
        model.Add(sum(assign[email, paper] for paper in paper_ids) >= min_load)

    # TODO: if the slack is greater than 1, create load elasticity variables
    #     load_elasticity_var = model.NewBoolVar(f'load_elasticity[{row["email"]}]')
    #     # load_elasticity_var is 1 if the load is exactly max_load or min_load
    #     model.Add(sum(assign[i, j] for j in range(num_papers)) < int(max_load) + load_elasticity_var)
    #     load_elasticity_vars.append(load_elasticity_var)

    # model.Add(sum(load_elasticity_vars) <= args.load_elasticity) if args.load_elasticity else None

    # account for HotCRP conflicts from topics_df
    # find entries in topics_df where 'conflict' column is 'conflict'
    conflict_entries = topics_df[topics_df['conflict'] == 'conflict']
    for index, row in conflict_entries.iterrows():
        email = row['email']
        paper = row['paper']
        model.Add(assign[email, paper] == 0)


    # account for reviews already assigned (say in R1 for R2)
    if pre_assigned_reviews_df is not None:
        for index, row in pre_assigned_reviews_df.iterrows():
            email = row['email']
            paper = row['paper']
            model.Add(assign[email, paper] == 1)

    # TODO: account for conflicted PCs assigned to the same paper

    # Create the objective function
    # Maximize the total score of the assignments
    objective_terms = []

    score = {}
    for email in pc_emails:
        for paper in paper_ids:
            score[email, paper] = 0

    for index, row in tpms_df.iterrows():
        email = row['email']
        paper = row['paper']
        tpms_score = row['tpms_score']
        score[email, paper] += tpms_score

    for index, row in topics_df.iterrows():
        email = row['email']
        paper = row['paper']
        topic_score = row['topic_score']
        preference = row['preference']
        score[email, paper] += topic_score
        score[email, paper] += preference

    for email in pc_emails:
        for paper in paper_ids:
            objective_terms.append(score[email, paper] * assign[email, paper])

    model.Maximize(sum(objective_terms))

    # Invoke the solver and print the solution
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status != cp_model.OPTIMAL:
        print_error(f'CpSolver returned status {CpSolver().StatusName(status)}')
        sys.exit(1)

    print(f'Optimal objective value: {solver.ObjectiveValue()}')

    assignment_df = pd.DataFrame(
        [{'email': e, 'paper': p} for e in pc_emails for p in paper_ids if solver.Value(assign[e, p]) == 1]
    )

    # drop those already in pre_assigned_reviews_df
    if pre_assigned_reviews_df is not None:
        assignment_df = assignment_df.merge(
            pre_assigned_reviews_df[['email', 'paper']], 
            on=['email', 'paper'], 
            how='left', indicator=True
        ).query('_merge == "left_only"').drop(columns='_merge')
    
    return assignment_df

def analyze_assignments(assignments_df, pcinfo_df, papers_df, topics_df, tpms_df, pre_assigned_reviews_df=None):
    load = assignments_df.groupby(['email'])['paper'].count().reset_index(name='count')
    load_count = load.groupby('count')['email'].count().reset_index(name='num_pc')
    # print the list of load_count and num_pc in a single line
    print('Load Counts (load, #pcs with this load): ' + ' '.join([f"{row['count']}:{row['num_pc']}" for index, row in load_count.iterrows()]))

    num_revs = assignments_df.groupby(['paper'])['email'].count().reset_index(name='count')
    num_revs_count = num_revs.groupby('count')['paper'].count().reset_index(name='num_papers')
    # print the list of load_count and num_papers in a single line
    print('Num Reviews (#revs, #papers with this #revs): ' + ' '.join([f"{row['count']}:{row['num_papers']}" for index, row in num_revs_count.iterrows()]))

    assignments_df = assignments_df.merge(topics_df, on=["email","paper"], how="left")
    num_prefs = assignments_df.groupby(['preference'])['paper'].count().reset_index(name='count')
    # print the list of num_prefs and count in a single line
    print('Num Preferences (#pref, #assignments with this #pref): ' + ' '.join([f"{row['preference']}:{row['count']}" for index, row in num_prefs.iterrows()]))

    merged = assignments_df \
        .merge(pcinfo_df, on="email", how="left") \
        .merge(papers_df, on="paper", how="left") \
        .merge(topics_df, on=["email","paper"], how="left") \
        .merge(tpms_df, on=["email","paper"], how="left")

    merged.to_csv(assignment_details_file, index=False)

def save_assignments(assignments_df, assignment_file):
    with open(assignment_file, 'w') as f:
        f.write('paper,assignment,email,round\n')
        for index, row in assignments_df.iterrows():
            f.write(f"{row['paper']},primaryreview,{row['email']},{round}\n")
    f.close()

main()



    

            
