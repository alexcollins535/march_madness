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
SEED = 77

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

    group = bracket_df[bracket_df[round_col] == game_id]

    teams = group['Team'].tolist()

    idx = teams.index(team)
    half = len(teams) // 2

    if idx < half:
        opponents = teams[half:]
    else:
        opponents = teams[:half]

    return opponents

#=====================================================
# Helper Functions
#=====================================================
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)

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

    return merged, team_to_idx_map, reformed

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
def compute_round_win_probabilities(team, round_col, bracket_df, team_to_idx_map, merged,  mc_model, feature_list, wins_model):

    opponents = get_possible_opponents(team, round_col, bracket_df)

    win_probs = []

    for opp in opponents:

        team1_series, team2_series = prep_team1_team2_series(
            team, opp, team_to_idx_map, merged
        )

        team_prob, opp_prob = compute_win_probas(
            team1_series, team2_series,
            mc_model, feature_list, wins_model
        )

        win_probs.append(team_prob)

    if len(win_probs) == 0:
        return None

    return {
        'opponents': opponents,
        'individual_probs': win_probs,
        'avg_win_prob': np.mean(win_probs),
        'median_win_prob': np.median(win_probs)
    }

def compute_team_round_outlook(team, bracket_df, team_to_idx_map, merged, mc_model, feature_list, wins_model, rounds=['1', '2', '3', '4']):

    round_strings = []
    wc_string = 'Round*_Game'
    for string in rounds:
        round_name = wc_string.replace('*', string)
        round_strings.append(round_name)

    # Purposefully skipping Rounds 5 and 6 (Final Four, limited merit in evaluating half the bracket)

    results = {}

    for i, r in enumerate(round_strings):
        print(f'[INFO] Computing Round {i+1}...')

        results[r] = compute_round_win_probabilities(
            team, r, bracket_df,
            team_to_idx_map, merged,
            mc_model, feature_list, wins_model
        )

    return results

def run_team_outlook_for_rounds(team1, rounds, merged, team_to_idx_map, mc_model, feature_list, wins_model, bracket_df):
    # Wrapper for compute_team_round_outlook
    results = compute_team_round_outlook(
        team1, bracket_df,
        team_to_idx_map, merged,
        mc_model, feature_list, wins_model,
        rounds
    )

    for round_name, info in results.items():
        if info is None:
            continue

        print(f'\n{round_name}')
        print('Possible Opponents:', info['opponents'])

        for opp, p in zip(info['opponents'], info['individual_probs']):
            print(f'  vs {opp}: {p:.3f}')

        print(f'Average Win Probability: {info['avg_win_prob']:.3f}')

#=====================================================
# BRACKET GENERATION
#=====================================================

ROUND_COLS = [
    'Round1_Game',
    'Round2_Game', 
    'Round3_Game',
    'Round4_Game',
    'Round5_Game',
    'Round6_Game'
]

HISTORICAL_UPSET_BUDGETS = None  # Will be computed from CSV

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

#=====================================================
# Main Routine
#=====================================================
CACHE_FILE = 'cache_files\mc_matchup_cache.pkl'
mc_cache = load_cache()


# Load all teams current data
print(f'[INFO] Loading data...')
merged, team_to_idx_map, bracket_df = load_data()

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

menu_string = ('=' * 60) + '\nChoose method:\n  1.) Manual bracket generation\n  2.) Automatic bracket generation\n  3.) Fill the cache\n' + ('=' * 60)
valid = False
while not valid:
    print(menu_string)
    method_selection = input('\nSelection: ')
    if method_selection in ['1', '2', '3']:
        valid = True
    else:
        print('Invalid input. Enter 1, 2, or 3.\n\n\n')

if method_selection == '1':
    # Request user input of team name or INI
    valid_names = team_to_idx_map.keys()
    valid = False
    while not valid:
        team1 = input('\nInput Team1: ')
        if team1 in valid_names:
            valid = True
        else:
            print('Invalid input. Use team name or INI.')

    menu_string = ('=' * 60) + '\nChoose routine:\n  1.) Choose Team2 for projection\n  2.) Run projections for all potential matchups\n  3.) Run projections for select rounds\n' + ('=' * 60)
    valid = False
    while not valid:
        print(menu_string)
        menu_selection = input('\nSelection: ')
        if menu_selection in ['1', '2', '3']:
            valid = True
        else:
            print('Invalid input. Enter 1, 2, or 3.\n\n\n')

    if menu_selection == '1':
        # choose team2 for 1v1 projection
        valid = False
        while not valid:
            team2 = input('\nInput Team2: ')
            if team2 in valid_names:
                valid = True
            else:
                print('Invalid input. Use team name or INI.')

        # Run with selected 1v1 matchup
        team1_series, team2_series = prep_team1_team2_series(team1, team2, team_to_idx_map, merged)
        team1_proba, team2_proba = compute_win_probas(team1_series, team2_series, mc_model, feature_list, wins_model)
        
        # Force the formatting to be Team not INI
        team1 = merged.iloc[team_to_idx_map[team1]]['Team']
        team2 = merged.iloc[team_to_idx_map[team2]]['Team']

        winner = team1 if team1_proba > team2_proba else team2

        print(f'{team1} win probability: {team1_proba:.3f}\n{team2} win probability: {team2_proba:.3f}\n\nModel favors {winner}.')

    elif menu_selection in ['2', '3']:
        if menu_selection == '2':
            rounds = ['1', '2', '3', '4']
        else:
            # Choose rounds
            valid = False
            while not valid:
                rounds_input = input('\nEvaluate potential matchups for round(s): ')
                rounds = rounds_input.split(', ')
                rounds = [x.strip() for x in rounds]

                bad_value = False
                for round in rounds:
                    if round not in ['1', '2', '3', '4']:
                        bad_value = True
                        break
                if not bad_value:
                    valid = True
                else:
                    print('Invalid input. Comma separated list can only contain values 1 through 4.\n\n\n')
        # Run with selected list of rounds
        run_team_outlook_for_rounds(team1, rounds, merged, team_to_idx_map, mc_model, feature_list, wins_model, bracket_df)

    save_cache(mc_cache)

elif method_selection == '2':
    # Compute historical upset budgets from past results
    print('[INFO] Computing historical upset budgets...')
    upset_budgets = compute_historical_budgets('historical_team_win_probabilities.csv')

    # Generate 25 diverse brackets
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

    # Export
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
