from overall_model_creation import overallMonteCarloModel
from mc_component_model_creation import add_opponent_seed
from joblib import load
import pandas as pd
import numpy as np
import warnings
import os
import hashlib
import pickle

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
SEED = 17

#=====================================================
# Helper Functions
#=====================================================
def backfill_2pt_stats(df):
    fg_pct = df['FG%']
    if fg_pct.max() > 1:
        fg_pct = fg_pct / 100

    missing_2pt_made = df['2PT MADE'].isna()
    missing_2pt_att = df['2PT ATT'].isna()

    df.loc[missing_2pt_made, '2PT MADE'] = (
        df.loc[missing_2pt_made, 'PPG']
        - df.loc[missing_2pt_made, 'FTPG']
        - 3 * df.loc[missing_2pt_made, '3PPG']
    ) / 2

    fg_made = df['2PT MADE'] + df['3PPG']
    fg_att = fg_made / fg_pct.replace(0, np.nan)

    df.loc[missing_2pt_att, '2PT ATT'] = (
        fg_att.loc[missing_2pt_att]
        - df.loc[missing_2pt_att, '3PTA']
    )

    df['2PT MADE'] = df['2PT MADE'].clip(lower=0)
    df['2PT ATT'] = df['2PT ATT'].clip(lower=0)

    return df

def prefix_stats(team_main, team_opponent, exclude):
    '''
    Returns a combined series where team_main stats are T1_ and opponent stats are T2_.
    '''
    main_prefixed = team_main.rename(lambda x: f'T1_{x}' if x not in exclude else x)
    opp_prefixed = team_opponent.rename(lambda x: f'T2_{x}' if x not in exclude else x)
    opp_prefixed = opp_prefixed.drop(exclude)

    combined = pd.concat([main_prefixed, opp_prefixed])
    return combined

def get_possible_opponents(team, round_col, bracket_df):

    team_row = bracket_df[bracket_df['Team'] == team].iloc[0]
    game_id = team_row[round_col]

    group = bracket_df[bracket_df[round_col] == game_id].copy()

    # Use Round1_Game to determine bracket position within the group
    # — this is stable regardless of round depth
    group = group.sort_values('Round1_Game')
    teams = group['Team'].tolist()

    team_idx = teams.index(team)
    half = len(teams) // 2

    if team_idx < half:
        opponents = teams[half:]
    else:
        opponents = teams[:half]

    return opponents

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)

def load_known_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_known_results(known_results):
    with open(RESULTS_FILE, 'wb') as f:
        pickle.dump(known_results, f)
    print(f'[INFO] Saved {len(known_results)} result(s) to {RESULTS_FILE}')

def hash_matchup(team1_series, team2_series):
    combined = pd.concat([team1_series, team2_series])
    combined.index = range(len(combined))  # force unique integer index
    data_bytes = combined.to_json().encode()
    return hashlib.sha256(data_bytes).hexdigest()

def canonicalize_matchup(team1_series, team2_series):

    if team1_series['Team'] < team2_series['Team']:
        return team1_series, team2_series, False
    else:
        return team2_series, team1_series, True

#=====================================================
# Compute Model Features
#=====================================================
def compute_win_probas(team1_series, team2_series, mc_model, feature_list, wins_model):

    # Copy raw inputs
    base_team1 = team1_series.copy()
    base_team2 = team2_series.copy()

    # Canonicalize matchup
    base_team1, base_team2, flipped = canonicalize_matchup(base_team1, base_team2)

    cache_key = hash_matchup(base_team1, base_team2)

    if cache_key in mc_cache:
        teamA_wins, teamB_wins = mc_cache[cache_key]
    else:
        teamA_wins, teamB_wins = mc_model.run_montecarlo_comparison(base_team1, base_team2, 1000)

        mc_cache[cache_key] = (teamA_wins, teamB_wins)

    if flipped:
        team1_wins = teamB_wins
        team2_wins = teamA_wins
    else:
        team1_wins = teamA_wins
        team2_wins = teamB_wins

    meta_win_dataset = []
    for team_series, team_wins, opp_series, opp_wins in [
        (team1_series, team1_wins, team2_series, team2_wins),
        (team2_series, team2_wins, team1_series, team1_wins)
    ]:
        mc_win_prob = team_wins / (team_wins + opp_wins)

        seed_diff = team_series['Seed'] - opp_series['Seed']

        ft_rate_diff = (
            team_series['T1_FTPG'] / team_series['T1_FGA']
            - team_series['T2_FTPG'] / team_series['T2_FGA']
        )
        three_pt_rate_diff = (
            team_series['T1_3PTA'] / team_series['T1_FGA']
            - team_series['T2_3PTA'] / team_series['T2_FGA']
        )

        avg_tpg = (team_series['T1_TPG'] + team_series['T2_TPG']) / 2
        avg_fb = (team_series['T1_FB'] + team_series['T2_FB']) / 2

        diff_sm = team_series['T1_SM'] - team_series['T2_SM']
        diff_ftpct = team_series['T1_FT%'] - team_series['T2_FT%']
        diff_fta = team_series['T1_FTA'] - team_series['T2_FTA']
        diff_3ppg = team_series['T1_3PPG'] - team_series['T2_3PPG']
        diff_tm = team_series['T1_TM'] - team_series['T2_TM']
        diff_tpg = team_series['T1_TPG'] - team_series['T2_TPG']

        meta_win_dataset.append({
                            'SEED_DIFF': seed_diff,
                            'MC_WIN_PROB': mc_win_prob,
                            'FT_RATE_DIFF': ft_rate_diff,
                            '3PT_RATE_DIFF': three_pt_rate_diff,
                            'AVG_TPG': avg_tpg,
                            'AVG_FB': avg_fb,
                            'DIFF_SM': diff_sm,
                            'DIFF_FT%': diff_ftpct,
                            'DIFF_FTA': diff_fta,
                            'DIFF_3PPG': diff_3ppg,
                            'DIFF_TM': diff_tm,
                            'DIFF_TPG': diff_tpg
                        })
        
    team1_series = pd.concat([team1_series, pd.Series(meta_win_dataset[0])])
    team2_series = pd.concat([team2_series, pd.Series(meta_win_dataset[1])])

    team1_features = pd.DataFrame([team1_series[feature_list].to_dict()])
    team2_features = pd.DataFrame([team2_series[feature_list].to_dict()])

    # Compute probabilities using the fitted pipeline
    team1_proba = wins_model.predict_proba(team1_features)[0, 1]
    team2_proba = wins_model.predict_proba(team2_features)[0, 1]

    team1_norm = team1_proba / (team1_proba + team2_proba)
    team2_norm = team2_proba / (team1_proba + team2_proba)
    return team1_norm, team2_norm

#=====================================================
# Initial Imports
#=====================================================
def load_models():
    structured_2025 = pd.read_excel('NCAA_Validation.xlsx', sheet_name='Structured 2025')
    structured_2024 = pd.read_excel('NCAA_Validation.xlsx', sheet_name='Structured 2024')
    structured_2023 = pd.read_excel('NCAA_Validation.xlsx', sheet_name='Structured 2023')

    structured_2025['YEAR'] = 2025
    structured_2024['YEAR'] = 2024
    structured_2023['YEAR'] = 2023

    combined = pd.concat([structured_2023, structured_2024, structured_2025])
    combined = backfill_2pt_stats(combined)
    combined = add_opponent_seed(combined)

    mc_model = overallMonteCarloModel(combined)
    wins_model_dict = load(r'mc_model_components\model_target_wins.joblib')
    wins_model = wins_model_dict['model']
    feature_list = wins_model_dict['feature_list']

    return mc_model, wins_model, feature_list

#=====================================================
# Extract current team info from spreadsheet
#=====================================================
def load_data():
    filepath = 'NCAA 2025-26.xlsx'
    main_df = pd.read_excel(filepath, sheet_name='BPI+')
    map_df = pd.read_excel(filepath, sheet_name='mapping')
    bracket_df = pd.read_excel(filepath, sheet_name='Bracket')

    df_pt1 = main_df[['Team*', 'Seed*']]
    df_pt2 = main_df.iloc[:, 101:133]
    df_pt2 = df_pt2.drop(['Blank', 'Res', 'BPI.1'], axis=1)
    df = pd.concat([df_pt1, df_pt2], axis=1)

    map_df = map_df[['BPI+ Value', 'INI']]

    merged = pd.merge(df, map_df, left_on='Team*', right_on='BPI+ Value', how='inner')
    merged = merged.drop(['BPI+ Value'], axis=1)
    merged.rename(columns={'Team*': 'Team'}, inplace=True)
    merged.rename(columns={'Seed*': 'Seed'}, inplace=True)
    merged.rename(columns={'Tempo': 'TEMPO'}, inplace=True)

    # Add missing stats - FGA, 2PT FGM, 2PT%
    new_features = pd.DataFrame({
            'FGA': 0.5 * merged['3PPG'] / (merged['EFG%'] - (merged['FG%'] / 100)),
            '2PT FGM': (merged['PPG'] - merged['FTPG'] - 3 * merged['3PPG']) / 2
    })
    merged = pd.concat([merged, new_features], axis=1)
    new_features = pd.DataFrame({
            '2PT%': merged['2PT FGM'] / (merged['FGA'] - merged['3PTA']) * 100,
    })
    merged = pd.concat([merged, new_features], axis=1)

    team_to_idx_map = {}
    for i, row in merged.iterrows():
        team_to_idx_map[row['INI']] = i
        team_to_idx_map[row['Team']] = i

    # Build a case-insensitive lookup that maps normalized keys -> original keys
    ci_team_lookup = {k.upper(): k for k in team_to_idx_map.keys()}

    # Process the bracket_df
    left_bracket = bracket_df[['Unnamed: 4', 'Unnamed: 2']]
    left_bracket.columns = ['Team', 'Seed']
    right_bracket = bracket_df[['Unnamed: 23', 'Unnamed: 25']]
    right_bracket.columns = ['Team', 'Seed']
    reformed = pd.concat([left_bracket, right_bracket], axis=0).dropna(subset=['Team']).reset_index(drop=True)

    # Store the games that a given team would participate in
    reformed['Round1_Game'] = (reformed.index // 2) + 1
    reformed['Round2_Game'] = (reformed.index // 4) + 33
    reformed['Round3_Game'] = (reformed.index // 8) + 49
    reformed['Round4_Game'] = (reformed.index // 16) + 57
    reformed['Round5_Game'] = (reformed.index // 32) + 61
    reformed['Round6_Game'] = (reformed.index // 64) + 63

    return merged, team_to_idx_map, reformed, ci_team_lookup

def resolve_team_input(raw_input, ci_team_lookup, team_to_idx_map):
    '''
    Accepts a team name or INI in any case.
    Returns the canonical key as stored in team_to_idx_map, or None if not found.
    '''
    return ci_team_lookup.get(raw_input.strip().upper())

def prep_team1_team2_series(team1, team2, team_to_idx_map, merged):
    team1_series = merged.iloc[team_to_idx_map[team1]]
    team2_series = merged.iloc[team_to_idx_map[team2]]

    exclude_cols = ['Team', 'INI', 'Seed']  # keep Seed unprefixed

    team1_view = prefix_stats(team1_series, team2_series, exclude=exclude_cols)
    team2_view = prefix_stats(team2_series, team1_series, exclude=exclude_cols)

    # Add Opp Seed manually (opponent's Seed from the other series)
    team1_view['Opp Seed'] = team2_series['Seed']
    team2_view['Opp Seed'] = team1_series['Seed']

    assert not team1_series.hasnans, f'{team1} series contains NaN values'
    assert not team2_series.hasnans, f'{team2} series contains NaN values'
    return team1_view, team2_view

def resolve_play_in_games(bracket_df, team_to_idx_map, merged,
                         mc_model, feature_list, wins_model):

    bracket_df = bracket_df.copy()

    for idx, row in bracket_df.iterrows():
        team_entry = row['Team']
        if isinstance(team_entry, str) and '/' in team_entry:
            # Example: 'BCU / HOW'
            t1, t2 = [t.strip() for t in team_entry.split('/')]

            if t1 not in team_to_idx_map or t2 not in team_to_idx_map:
                print(f'[WARNING] Missing mapping for at least one of {t1}, {t2}. Skipping.')
                continue

            # Prepare model inputs
            team1_series, team2_series = prep_team1_team2_series(t1, t2, team_to_idx_map, merged)
            p1, p2 = compute_win_probas(team1_series, team2_series, mc_model, feature_list, wins_model)

            winner_ini = t1 if p1 > p2 else t2
            winner = merged.iloc[team_to_idx_map[winner_ini]]['Team']

            print(f'Play-in: {t1} vs {t2} -> {winner} ({max(p1,p2):.3f})')

            bracket_df.at[idx, 'Team'] = winner

    return bracket_df

#=====================================================
# Functions for team outlook by round
#=====================================================
def compute_reach_probability(team, round_num, bracket_df, team_to_idx_map, merged,
                               mc_model, feature_list, wins_model, known_results=None):
    '''
    P(team reaches round_num).
    If a known result exists for a prior game, uses that instead of probabilities.
    '''
    if known_results is None:
        known_results = {}

    if round_num == 1:
        return 1.0

    # Check if we know the result of the game that determined entry into round_num
    prev_round_num = round_num - 1
    prev_round_col = f'Round{prev_round_num}_Game'
    team_row = bracket_df[bracket_df['Team'] == team].iloc[0]
    prev_game_id = team_row[prev_round_col]

    if (prev_round_num, prev_game_id) in known_results:
        actual_winner = known_results[(prev_round_num, prev_game_id)]
        if actual_winner == team:
            # Team won this game — now check if they reached the round before that
            return compute_reach_probability(
                team, prev_round_num, bracket_df, team_to_idx_map, merged,
                mc_model, feature_list, wins_model, known_results
            )
        else:
            # Team lost — they cannot reach round_num
            return 0.0

    # No known result — compute probabilistically
    p_team_reached_prev = compute_reach_probability(
        team, prev_round_num, bracket_df, team_to_idx_map, merged,
        mc_model, feature_list, wins_model, known_results
    )

    prev_opponents = get_possible_opponents(team, prev_round_col, bracket_df)

    p_win_prev_round = 0.0
    for opp in prev_opponents:
        # If we know this opponent lost already, their reach prob is 0
        p_opp_reached = compute_reach_probability(
            opp, prev_round_num, bracket_df, team_to_idx_map, merged,
            mc_model, feature_list, wins_model, known_results
        )

        t1_series, t2_series = prep_team1_team2_series(
            team, opp, team_to_idx_map, merged
        )
        p_beat_opp, _ = compute_win_probas(
            t1_series, t2_series,
            mc_model, feature_list, wins_model
        )
        p_win_prev_round += p_beat_opp * p_opp_reached

    return p_team_reached_prev * p_win_prev_round

def compute_round_win_probabilities(team, round_col, bracket_df, team_to_idx_map, merged,
                                     mc_model, feature_list, wins_model, known_results=None):
    if known_results is None:
        known_results = {}

    round_num = int(round_col.replace('Round', '').replace('_Game', ''))
    opponents = get_possible_opponents(team, round_col, bracket_df)

    p_team_reached = compute_reach_probability(
        team, round_num, bracket_df, team_to_idx_map, merged,
        mc_model, feature_list, wins_model, known_results
    )

    win_probs = []
    reach_probs = []

    for opp in opponents:
        t1_series, t2_series = prep_team1_team2_series(
            team, opp, team_to_idx_map, merged
        )
        p_beat_opp, _ = compute_win_probas(
            t1_series, t2_series,
            mc_model, feature_list, wins_model
        )
        p_opp_reached = compute_reach_probability(
            opp, round_num, bracket_df, team_to_idx_map, merged,
            mc_model, feature_list, wins_model, known_results
        )
        win_probs.append(p_beat_opp)
        reach_probs.append(p_opp_reached)

    if not win_probs:
        return None

    weighted_win_prob = p_team_reached * sum(
        p_win * p_reach for p_win, p_reach in zip(win_probs, reach_probs)
    )

    return {
        'opponents': opponents,
        'individual_probs': win_probs,
        'reach_probs': reach_probs,
        'p_team_reached': p_team_reached,
        'weighted_win_prob': weighted_win_prob,
        'reach_probs_sum': sum(reach_probs)
    }

def compute_team_round_outlook(team, bracket_df, team_to_idx_map, merged, mc_model,
                                feature_list, wins_model, rounds=['1', '2', '3', '4'],
                                known_results=None):
    if known_results is None:
        known_results = {}

    round_strings = [f'Round{r}_Game' for r in rounds]
    results = {}

    for i, r in enumerate(round_strings):
        print(f'[INFO] Computing Round {rounds[i]}...')
        results[r] = compute_round_win_probabilities(
            team, r, bracket_df,
            team_to_idx_map, merged,
            mc_model, feature_list, wins_model,
            known_results
        )

    return results

def run_team_outlook_for_rounds(team1, rounds, merged, team_to_idx_map, mc_model,
                                 feature_list, wins_model, bracket_df, known_results=None):
    if known_results is None:
        known_results = {}

    results = compute_team_round_outlook(
        team1, bracket_df,
        team_to_idx_map, merged,
        mc_model, feature_list, wins_model,
        rounds, known_results
    )

    for round_name, info in results.items():
        if info is None:
            continue

        print(f'\n{round_name}')
        print(f'P({team1} reaches this round): {info["p_team_reached"]:.3f}')
        print('Possible Opponents:', info['opponents'])

        for opp, p_win, p_reach in zip(info['opponents'], info['individual_probs'], info['reach_probs']):
            print(f'  vs {opp}: win prob = {p_win:.3f}, opp reach prob = {p_reach:.3f}')

        print(f'Weighted Win Probability: {info["weighted_win_prob"]:.3f}')

def compute_round_matchup_probabilities(round_num, bracket_df, team_to_idx_map, merged,
                                         mc_model, feature_list, wins_model,
                                         prior_winners=None):
    '''
    Print a sorted table of win probabilities for every game in a given round.
    For rounds > 1, prior_winners is a dict of {game_id: winner_name} for all
    games in the preceding round, used to advance teams before computing matchups.
    '''
    round_col = f'Round{round_num}_Game'
    prev_round_col = f'Round{round_num - 1}_Game' if round_num > 1 else None

    active_bracket = bracket_df.copy()

    # Advance bracket using prior round results
    if prior_winners and prev_round_col:
        surviving_teams = list(prior_winners.values())
        active_bracket = active_bracket[
            active_bracket['Team'].isin(surviving_teams)
        ].copy()

    game_ids = active_bracket[round_col].dropna().unique()

    results = []

    for game_id in game_ids:
        participants = active_bracket[
            active_bracket[round_col] == game_id
        ]['Team'].tolist()

        if len(participants) != 2:
            print(f'[WARNING] Game {game_id} has {len(participants)} participant(s), skipping.')
            continue

        t1, t2 = participants[0], participants[1]

        if t1 not in team_to_idx_map or t2 not in team_to_idx_map:
            print(f'[WARNING] Missing mapping for {t1} or {t2}, skipping.')
            continue

        t1_series, t2_series = prep_team1_team2_series(t1, t2, team_to_idx_map, merged)
        p1, p2 = compute_win_probas(t1_series, t2_series, mc_model, feature_list, wins_model)

        favorite  = t1 if p1 >= p2 else t2
        underdog  = t2 if p1 >= p2 else t1
        fav_prob  = max(p1, p2)
        und_prob  = min(p1, p2)

        results.append({
            'game_id':  game_id,
            'favorite': favorite,
            'fav_prob': fav_prob,
            'underdog': underdog,
            'und_prob': und_prob,
        })

    # Sort highest-confidence (largest fav_prob) first
    results.sort(key=lambda x: x['fav_prob'], reverse=True)

    # Print table
    header = f'{'Game ID':<10}{'Favorite':<30}{'Fav %':<10}{'Underdog':<30}{'Und %':<10}'
    print(f'\nRound {round_num} Matchup Probabilities')
    print('=' * len(header))
    print(header)
    print('-' * len(header))
    for r in results:
        print(
            f'{r['game_id']:<10}'
            f'{r['favorite']:<30}'
            f'{r['fav_prob']:.3f}     '
            f'{r['underdog']:<30}'
            f'{r['und_prob']:.3f}'
        )
    print('=' * len(header))

def prompt_prior_round_winners(round_num, bracket_df, team_to_idx_map):
    '''
    For rounds > 1, prompt the user to enter the winner of each game
    in the preceding round. Returns a dict of {game_id: winner_team_name}.
    '''
    prev_round_col = f'Round{round_num - 1}_Game'
    game_ids = bracket_df[prev_round_col].dropna().unique()

    prior_winners = {}
    valid_names = set(team_to_idx_map.keys())

    print(f'\nEnter the winner of each Round {round_num - 1} game:')

    for game_id in sorted(game_ids):
        participants = bracket_df[
            bracket_df[prev_round_col] == game_id
        ]['Team'].tolist()

        prompt = f'  Game {game_id} ({' vs '.join(participants)}): '

        valid = False
        while not valid:
            winner = input(prompt).strip()
            resolved = resolve_team_input(winner, ci_team_lookup, team_to_idx_map)
            if resolved is not None:
                full_name = merged.iloc[team_to_idx_map[resolved]]['Team']
                prior_winners[game_id] = full_name
                valid = True
            else:
                print(f'Invalid input. Use team name or INI. Options: {participants}')

    return prior_winners

def prompt_known_results(bracket_df, team_to_idx_map, ci_team_lookup, merged):
    '''
    Prompts the user to enter actual game results that have already occurred.
    Starts from whatever is already saved so only new results need to be entered.
    Returns a dict of {(round_num, game_id): winner_team_name}.
    '''
    # Start from whatever is already saved so the user only adds new results
    known_results = load_known_results()

    print('\nEnter known results. Press Enter to skip a game (result unknown).')

    for round_num, round_col in enumerate(ROUND_COLS, start=1):
        game_ids = bracket_df[round_col].dropna().unique()
        round_complete = True

        for game_id in sorted(game_ids):
            participants = bracket_df[
                bracket_df[round_col] == game_id
            ]['Team'].tolist()

            if len(participants) != 2:
                continue

            # Show existing result if already recorded
            existing = known_results.get((round_num, game_id))
            existing_str = f' [current: {existing}]' if existing else ''
            prompt = f'  Round {round_num} Game {game_id} ({" vs ".join(participants)}){existing_str}: '

            while True:
                winner_input = input(prompt).strip()

                if winner_input == '':
                    # Keep existing result if present, otherwise mark incomplete
                    if not existing:
                        round_complete = False
                    break

                resolved = resolve_team_input(winner_input, ci_team_lookup, team_to_idx_map)
                if resolved is not None:
                    full_name = merged.iloc[team_to_idx_map[resolved]]['Team']
                    if full_name in participants:
                        known_results[(round_num, game_id)] = full_name
                        break
                    else:
                        print(f'    {full_name} is not a participant in this game. Options: {participants}')
                else:
                    print(f'    Invalid input. Use team name or INI. Options: {participants}')

        if not round_complete:
            break

    print(f'\n[INFO] {len(known_results)} result(s) recorded.')
    return known_results

#=====================================================
# BRACKET GENERATION
#=====================================================
def compute_historical_budgets(csv_path='historical_team_win_probabilities.csv'):
    df = pd.read_csv(csv_path)

    def assign_round(game_id):
        if game_id <= 4:    return 0
        elif game_id <= 36: return 1
        elif game_id <= 52: return 2
        elif game_id <= 60: return 3
        elif game_id <= 64: return 4
        elif game_id <= 66: return 5
        else:               return 6

    df['Round'] = df['Game ID'].apply(assign_round)
    df = df[df['Round'] > 0]

    budgets_by_round = {}
    for round_num in range(1, 7):
        round_df = df[df['Round'] == round_num]

        yearly_budgets = []
        for year in round_df['Year'].unique():
            year_games = round_df[round_df['Year'] == year]

            total_upset_mass = 0.0
            for game_id in year_games['Game ID'].unique():
                game_rows = year_games[year_games['Game ID'] == game_id]
                if len(game_rows) < 2:
                    continue

                winner_row = game_rows[game_rows['Points Scored'] > game_rows['Opponent Points']]
                if len(winner_row) == 0:
                    continue

                winner_prob = winner_row.iloc[0]['Pred Win Prob']

                # Only count actual upsets (winner was not the model favorite)
                if winner_prob < 0.5:
                    total_upset_mass += (1 - winner_prob)  # cost = how unlikely this upset was

            yearly_budgets.append(total_upset_mass)

        budgets_by_round[round_num] = np.mean(yearly_budgets)
        print(f'[INFO] Round {round_num} avg upset budget: {budgets_by_round[round_num]:.3f}')

    return budgets_by_round

def simulate_bracket(bracket_df, merged, team_to_idx_map,
                     mc_model, feature_list, wins_model,
                     upset_budgets, rng):

    active_bracket = bracket_df.copy()
    picks = {}
    favorites = {}

    for round_num, round_col in enumerate(ROUND_COLS, start=1):
        if round_col not in active_bracket.columns:
            break

        budget = upset_budgets.get(round_num, 0.0)
        remaining_budget = budget

        game_ids = active_bracket[round_col].dropna().unique()

        # Compute win probabilities for all games this round
        game_probs = {}
        for game_id in game_ids:
            participants = active_bracket[
                active_bracket[round_col] == game_id
            ]['Team'].tolist()

            if len(participants) != 2:
                continue

            t1, t2 = participants[0], participants[1]
            if t1 not in team_to_idx_map or t2 not in team_to_idx_map:
                continue

            t1_series, t2_series = prep_team1_team2_series(
                t1, t2, team_to_idx_map, merged
            )
            p1, p2 = compute_win_probas(
                t1_series, t2_series,
                mc_model, feature_list, wins_model
            )

            favorite  = t1 if p1 >= p2 else t2
            underdog  = t2 if p1 >= p2 else t1
            upset_prob = min(p1, p2)       # probability the underdog wins
            upset_cost = 1 - upset_prob    # how surprising the upset would be

            game_probs[game_id] = {
                'team1': t1, 'team2': t2,
                'p1': p1, 'p2': p2,
                'favorite': favorite,
                'underdog': underdog,
                'upset_prob': upset_prob,
                'upset_cost': upset_cost
            }

        # Shuffle matchup order so no game has systematic priority
        shuffled_games = list(game_probs.items())
        rng.shuffle(shuffled_games)

        round_winners = {}
        for game_id, info in shuffled_games:
            favorites[(round_num, game_id)] = info['favorite']

            if remaining_budget > 0:
                draw = rng.uniform(0, 1)
                if draw < info['upset_prob']:
                    # Upset occurs - deduct surprise cost from budget
                    winner = info['underdog']
                    remaining_budget -= info['upset_cost']
                else:
                    winner = info['favorite']
            else:
                # Budget exhausted - take model favorite for all remaining games
                winner = info['favorite']

            round_winners[game_id] = winner
            picks[(round_num, game_id)] = winner

        # Advance winners to next round
        if round_num < len(ROUND_COLS):
            surviving_teams = list(round_winners.values())
            active_bracket = active_bracket[
                active_bracket['Team'].isin(surviving_teams)
            ].copy()

    return picks, favorites

def generate_diverse_brackets(n_brackets, bracket_df, merged, team_to_idx_map,
                               mc_model, feature_list, wins_model,
                               upset_budgets):
    '''
    Generate n_brackets diverse bracket picks by iteratively penalising
    games that have already been chosen as upsets in previous brackets.
    '''
    rng = np.random.default_rng(SEED)

    all_brackets = []


    for i in range(n_brackets):
        print(f'[INFO] Generating bracket {i + 1}/{n_brackets}...')

        picks, favorites = simulate_bracket(
            bracket_df, merged, team_to_idx_map,
            mc_model, feature_list, wins_model,
            upset_budgets, rng
        )

        all_brackets.append((picks, favorites))

    return all_brackets

def export_brackets(all_brackets, bracket_df, filepath='generated_brackets.csv'):
    '''
    Export all brackets to a flat CSV with columns:
    Bracket, Round, Game_ID, Winner
    '''
    rows = []
    for i, (picks, favorites) in enumerate(all_brackets):
        for (round_num, game_id), winner in picks.items():
            favorite = favorites.get((round_num, game_id), None)
            rows.append({
                'Bracket': i + 1,
                'Round': round_num,
                'Game_ID': game_id,
                'Winner': winner,
                'Model_Favorite': favorite,
                'Is_Upset': winner != favorite
            })
    df = pd.DataFrame(rows).sort_values(['Bracket', 'Round', 'Game_ID'])
    df.to_csv(filepath, index=False)
    print(f'[INFO] Exported {len(all_brackets)} brackets to {filepath}')
    return df

def precompute_quarter_matchup_cache(bracket_df, merged, team_to_idx_map,
                                      mc_model, feature_list, wins_model):
    '''
    For every team in the bracket, find all 16 teams in their quarter
    (Round4_Game group), then run mc simulations for every unique pairwise
    matchup within that quarter. Results land in mc_cache automatically.
    Saves mc_cache to disk after each quarter is processed.
    '''
    # Get unique quarters (each Round4_Game group = 16 teams)
    quarters = bracket_df['Round4_Game'].dropna().unique()
    total_quarters = len(quarters)

    for q_idx, quarter_id in enumerate(quarters):
        quarter_teams = bracket_df[
            bracket_df['Round4_Game'] == quarter_id
        ]['Team'].tolist()

        valid_teams = [t for t in quarter_teams if t in team_to_idx_map]

        print(f'\n[INFO] Quarter {q_idx+1}/{total_quarters} '
              f'(Game ID {quarter_id}) — {len(valid_teams)} teams')

        # All unique pairs within the quarter
        pairs = [
            (valid_teams[i], valid_teams[j])
            for i in range(len(valid_teams))
            for j in range(i + 1, len(valid_teams))
        ]

        cached = 0
        computed = 0

        for t1, t2 in pairs:
            print(f'{t1} vs. {t2}')

            t1_series, t2_series = prep_team1_team2_series(
                t1, t2, team_to_idx_map, merged
            )

            # Canonicalize to match how mc_cache keys are generated
            base_t1, base_t2, _ = canonicalize_matchup(
                merged.iloc[team_to_idx_map[t1]].copy(),
                merged.iloc[team_to_idx_map[t2]].copy()
            )
            cache_key = hash_matchup(base_t1, base_t2)

            if cache_key in mc_cache:
                cached += 1
                continue

            # This call populates mc_cache internally
            compute_win_probas(
                t1_series, t2_series,
                mc_model, feature_list, wins_model
            )
            computed += 1

        save_cache(mc_cache)
        print(f'[INFO] Quarter {q_idx+1} done — '
              f'{computed} computed, {cached} already cached. '
              f'mc_cache size: {len(mc_cache)}')

    print(f'\n[INFO] All quarters precomputed. '
          f'Total mc_cache entries: {len(mc_cache)}')

def print_matchup_features(team1, team2, team_to_idx_map, merged,
                            mc_model, feature_list, wins_model):
    '''
    Prints the feature values used by the model for a specific matchup,
    from team1's perspective.
    '''
    t1_series, t2_series = prep_team1_team2_series(team1, team2, team_to_idx_map, merged)

    # Canonicalize using raw rows (for cache key consistency with compute_win_probas)
    base_t1_raw = merged.iloc[team_to_idx_map[team1]].copy()
    base_t2_raw = merged.iloc[team_to_idx_map[team2]].copy()
    base_t1_raw, base_t2_raw, flipped = canonicalize_matchup(base_t1_raw, base_t2_raw)
    cache_key = hash_matchup(base_t1_raw, base_t2_raw)

    # Canonicalize prefixed series to match
    if flipped:
        base_t1_prefixed, base_t2_prefixed = t2_series.copy(), t1_series.copy()
    else:
        base_t1_prefixed, base_t2_prefixed = t1_series.copy(), t2_series.copy()

    if cache_key in mc_cache:
        teamA_wins, teamB_wins = mc_cache[cache_key]
    else:
        teamA_wins, teamB_wins = mc_model.run_montecarlo_comparison(base_t1_prefixed, base_t2_prefixed, 1000)
        mc_cache[cache_key] = (teamA_wins, teamB_wins)

    team1_wins = teamB_wins if flipped else teamA_wins
    team2_wins = teamA_wins if flipped else teamB_wins

    # Build meta features for both perspectives
    def build_meta(team_series, team_wins, opp_wins, seed, opp_seed):
        mc_win_prob = team_wins / (team_wins + opp_wins)
        return {
            'SEED_DIFF':      seed - opp_seed,
            'MC_WIN_PROB':    mc_win_prob,
            'FT_RATE_DIFF':   team_series['T1_FTPG'] / team_series['T1_FGA'] - team_series['T2_FTPG'] / team_series['T2_FGA'],
            '3PT_RATE_DIFF':  team_series['T1_3PTA'] / team_series['T1_FGA'] - team_series['T2_3PTA'] / team_series['T2_FGA'],
            'AVG_TPG':        (team_series['T1_TPG'] + team_series['T2_TPG']) / 2,
            'AVG_FB':         (team_series['T1_FB']  + team_series['T2_FB'])  / 2,
            'DIFF_SM':        team_series['T1_SM']   - team_series['T2_SM'],
            'DIFF_FT%':       team_series['T1_FT%']  - team_series['T2_FT%'],
            'DIFF_FTA':       team_series['T1_FTA']  - team_series['T2_FTA'],
            'DIFF_3PPG':      team_series['T1_3PPG'] - team_series['T2_3PPG'],
            'DIFF_TM':        team_series['T1_TM']   - team_series['T2_TM'],
            'DIFF_TPG':       team_series['T1_TPG']  - team_series['T2_TPG'],
        }

    t1_meta = build_meta(t1_series, team1_wins, team2_wins,
                         merged.iloc[team_to_idx_map[team1]]['Seed'],
                         merged.iloc[team_to_idx_map[team2]]['Seed'])
    t2_meta = build_meta(t2_series, team2_wins, team1_wins,
                         merged.iloc[team_to_idx_map[team2]]['Seed'],
                         merged.iloc[team_to_idx_map[team1]]['Seed'])

    t1_full = pd.concat([t1_series, pd.Series(t1_meta)])
    t2_full = pd.concat([t2_series, pd.Series(t2_meta)])

    t1_features = pd.DataFrame([t1_full[feature_list].to_dict()])
    t2_features = pd.DataFrame([t2_full[feature_list].to_dict()])

    t1_proba = wins_model.predict_proba(t1_features)[0, 1]
    t2_proba = wins_model.predict_proba(t2_features)[0, 1]
    t1_norm = t1_proba / (t1_proba + t2_proba)
    t2_norm = t2_proba / (t1_proba + t2_proba)

    # Print
    col_w = 32
    header = f'  {"Feature":<25} {team1:>{col_w}} {team2:>{col_w}}'
    print(f'\nMatchup Features: {team1} vs {team2}')
    print('=' * len(header))
    print(header)
    print('-' * len(header))
    for feat in feature_list:
        v1 = t1_features[feat].iloc[0]
        v2 = t2_features[feat].iloc[0]
        print(f'  {feat:<25} {v1:>{col_w}.4f} {v2:>{col_w}.4f}')
    print('=' * len(header))
    print(f'  {"Win Probability":<25} {t1_norm:>{col_w}.3f} {t2_norm:>{col_w}.3f}')
    print('=' * len(header))

#=====================================================
# Main Routine
#=====================================================
CACHE_FILE = r'cache_files\mc_matchup_cache.pkl'
ROUND_COLS = [
    'Round1_Game',
    'Round2_Game', 
    'Round3_Game',
    'Round4_Game',
    'Round5_Game',
    'Round6_Game'
]
HISTORICAL_UPSET_BUDGETS = None  # Will be computed from CSV
RESULTS_FILE = r'cache_files\known_results.pkl'

mc_cache = load_cache()
known_results = load_known_results()

# Load all teams current data
print(f'[INFO] Loading data...')
merged, team_to_idx_map, bracket_df, ci_team_lookup = load_data()

# Load models
mc_model, wins_model, feature_list = load_models()

# Resolve play-in games
print(f'[INFO] Resolving play-in games...')
bracket_df = resolve_play_in_games(
    bracket_df,
    team_to_idx_map,
    merged,
    mc_model,
    feature_list,
    wins_model
)

menu_string = ('=' * 60) + '\nChoose routine:\n  1.) Manual bracket generation\n  2.) Automatic bracket generation\n  3.) Fill the cache\n  4.) Input known results\n' + ('=' * 60)
valid = False
while not valid:
    print(menu_string)
    method_selection = input('\nSelection: ')
    if method_selection in ['1', '2', '3', '4']:
        valid = True
    else:
        print('Invalid input. Enter 1, 2, 3, or 4.\n\n\n')

if method_selection == '1':
    valid_names = set(team_to_idx_map.keys())

    valid = False
    while not valid:
        teams_input = input('\nInput Team(s) (comma separated): ')
        raw_teams = [t.strip() for t in teams_input.split(',')]

        resolved_teams = []
        bad_input = False
        for raw in raw_teams:
            resolved = resolve_team_input(raw, ci_team_lookup, team_to_idx_map)
            if resolved is None:
                print(f'Invalid input: {raw}. Use team name or INI.')
                bad_input = True
                break
            resolved_teams.append(merged.iloc[team_to_idx_map[resolved]]['Team'])

        if not bad_input:
            valid = True

    menu_string = ('=' * 60) + '\nChoose routine:\n  1.) Choose Team2 for projection\n  2.) Run projections for all potential matchups\n  3.) Run projections for select rounds\n  4.) Print matchup features\n' + ('=' * 60)
    valid = False
    while not valid:
        print(menu_string)
        menu_selection = input('\nSelection: ')
        if menu_selection in ['1', '2', '3', '4']:
            valid = True
        else:
            print('Invalid input. Enter 1, 2, 3, or 4.\n\n\n')

    if menu_selection == '1':
        if len(resolved_teams) > 1:
            print('Option 1 requires exactly one team. Using first team:', resolved_teams[0])
        team1 = resolved_teams[0]

        valid = False
        while not valid:
            team2 = input('\nInput Team2: ')
            team2 = resolve_team_input(team2, ci_team_lookup, team_to_idx_map)
            if team2 is not None:
                valid = True
            else:
                print('Invalid input. Use team name or INI.')
        team2 = merged.iloc[team_to_idx_map[team2]]['Team']

        team1_series, team2_series = prep_team1_team2_series(team1, team2, team_to_idx_map, merged)
        team1_proba, team2_proba = compute_win_probas(team1_series, team2_series, mc_model, feature_list, wins_model)

        winner = team1 if team1_proba > team2_proba else team2
        print(f'{team1} win probability: {team1_proba:.3f}\n{team2} win probability: {team2_proba:.3f}\n\nModel favors {winner}.')

    elif menu_selection in ['2', '3']:
        if menu_selection == '2':
            rounds = ['1', '2', '3', '4']
        else:
            valid = False
            while not valid:
                rounds_input = input('\nEvaluate potential matchups for round(s): ')
                rounds = [x.strip() for x in rounds_input.split(',')]
                if all(r in ['1', '2', '3', '4'] for r in rounds):
                    valid = True
                else:
                    print('Invalid input. Comma separated list can only contain values 1 through 4.\n\n\n')

        # Optionally collect known results
        print(f'\n[INFO] {len(known_results)} known result(s) currently loaded.')
        use_results = input('Update known results? (y/n): ').strip().lower()
        if use_results == 'y':
            known_results = prompt_known_results(bracket_df, team_to_idx_map, ci_team_lookup, merged)
            save_known_results(known_results)

        for team1 in resolved_teams:
            print(f'\n{"=" * 60}\nOutlook for {team1}\n{"=" * 60}')
            run_team_outlook_for_rounds(team1, rounds, merged, team_to_idx_map, mc_model,
                                         feature_list, wins_model, bracket_df, known_results)

    elif menu_selection == '4':
        if len(resolved_teams) > 1:
            print('Option 4 requires exactly one team. Using first team:', resolved_teams[0])
        team1 = resolved_teams[0]

        valid = False
        while not valid:
            team2 = input('\nInput Team2: ')
            team2 = resolve_team_input(team2, ci_team_lookup, team_to_idx_map)
            if team2 is not None:
                valid = True
            else:
                print('Invalid input. Use team name or INI.')
        team2 = merged.iloc[team_to_idx_map[team2]]['Team']

        print_matchup_features(team1, team2, team_to_idx_map, merged,
                                mc_model, feature_list, wins_model)

    save_cache(mc_cache)



elif method_selection == '2':
    # Compute historical upset budgets from past results
    print('[INFO] Computing historical upset budgets...')
    upset_budgets = compute_historical_budgets('historical_team_win_probabilities.csv')

    # Generate diverse brackets
    print('[INFO] Generating brackets...')
    all_brackets = generate_diverse_brackets(
        n_brackets=10,
        bracket_df=bracket_df,
        merged=merged,
        team_to_idx_map=team_to_idx_map,
        mc_model=mc_model,
        feature_list=feature_list,
        wins_model=wins_model,
        upset_budgets=upset_budgets
    )

    export_brackets(all_brackets, bracket_df)
    save_cache(mc_cache)

elif method_selection == '3':
    print('[INFO] Precomputing mc simulations for all quarter matchups...')
    precompute_quarter_matchup_cache(
        bracket_df=bracket_df,
        merged=merged,
        team_to_idx_map=team_to_idx_map,
        mc_model=mc_model,
        feature_list=feature_list,
        wins_model=wins_model
    )
    save_cache(mc_cache)

elif method_selection == '4':
    results_menu_string = ('=' * 60) + '\nManage known results:\n  1.) View current results\n  2.) Update results\n  3.) Clear all results\n' + ('=' * 60)
    valid = False
    while not valid:
        print(results_menu_string)
        results_selection = input('\nSelection: ')
        if results_selection in ['1', '2', '3']:
            valid = True
        else:
            print('Invalid input. Enter 1, 2, or 3.\n\n\n')

    if results_selection == '1':
        if not known_results:
            print('\nNo results currently recorded.')
        else:
            print(f'\n{len(known_results)} result(s) on record:')
            for (round_num, game_id), winner in sorted(known_results.items()):
                print(f'  Round {round_num} Game {game_id}: {winner}')

    elif results_selection == '2':
        known_results = prompt_known_results(bracket_df, team_to_idx_map, ci_team_lookup, merged)
        save_known_results(known_results)

    elif results_selection == '3':
        confirm = input('\nAre you sure you want to clear all results? (y/n): ').strip().lower()
        if confirm == 'y':
            known_results = {}
            save_known_results(known_results)
            print('[INFO] All results cleared.')