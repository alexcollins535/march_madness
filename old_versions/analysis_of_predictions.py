import pandas as pd
import numpy as np


def get_accuracy(series, for_print=True):
    acc = np.mean(series)
    return np.round(acc * 100, 2) if for_print else acc


df1 = pd.read_csv('tournament_projections.csv')
df2 = pd.read_csv('tournament_projections_v2.csv')
df3 = pd.read_csv('tournament_projections_v3.csv')

for df in [df3]:

    # --------------------------------------------------
    # Identify opponent rows
    # --------------------------------------------------
    df['opponent_seed'] = np.where(
        df.index % 2 == 0,
        df['team_seed'].shift(-1),
        df['team_seed'].shift(1)
    ).astype(int)

    df['opponent_name'] = np.where(
        df.index % 2 == 0,
        df['team_name'].shift(-1),
        df['team_name'].shift(1)
    )

    # --------------------------------------------------
    # Confidence proxy (unchanged)
    # --------------------------------------------------
    df['abs_diff'] = np.abs(
        df['projected_wins_team1'] - df['projected_wins_team2']
    )

    # --------------------------------------------------
    # Actual outcome
    # --------------------------------------------------
    df['team_won'] = df['team_score'] > df['opponent_score']

    # --------------------------------------------------
    # Model prediction
    # --------------------------------------------------
    df['model_pick'] = df['projected_wins_team1'] > df['projected_wins_team2']
    df['model_correct'] = df['model_pick'] == df['team_won']

    # --------------------------------------------------
    # Chalk prediction (lower seed wins)
    # --------------------------------------------------
    df['chalk_pick'] = df['team_seed'] < df['opponent_seed']
    df['chalk_correct'] = df['chalk_pick'] == df['team_won']

    # --------------------------------------------------
    # First round flag
    # --------------------------------------------------
    df['first_round'] = (df['team_seed'] + df['opponent_seed']) == 17

    # --------------------------------------------------
    # Filter: only different-seed matchups
    # --------------------------------------------------
    diff_seed_df = df[df['team_seed'] != df['opponent_seed']]
    first_round_diff_seed = diff_seed_df[diff_seed_df['first_round']]

    # ==================================================
    # OVERALL ACCURACY
    # ==================================================
    print("OVERALL (DIFF SEEDS ONLY)")
    print(f"Model Accuracy : {get_accuracy(diff_seed_df['model_correct'])}%")
    print(f"Chalk Accuracy : {get_accuracy(diff_seed_df['chalk_correct'])}%\n")

    # ==================================================
    # FIRST ROUND ACCURACY
    # ==================================================
    print("FIRST ROUND (DIFF SEEDS ONLY)")
    print(f"Model Accuracy : {get_accuracy(first_round_diff_seed['model_correct'])}%")
    print(f"Chalk Accuracy : {get_accuracy(first_round_diff_seed['chalk_correct'])}%\n")

    print("OVERALL (ALL INCLUSIVE)")
    print(f"Model Accuracy : {get_accuracy(df['model_correct'])}%")
   
    # ==================================================
    # CONFIDENCE STRATIFICATION (DIFF SEEDS ONLY)
    # ==================================================
    confidence_bins = [
        (0, 100),
        (100, 200),
        (200, 300),
        (300, 400),
        (400, np.inf)
    ]

    print("CONFIDENCE STRATA (MODEL vs CHALK)")
    for low, high in confidence_bins:

        bucket = diff_seed_df[
            (diff_seed_df['abs_diff'] >= low) &
            (diff_seed_df['abs_diff'] < high)
        ]

        if len(bucket) == 0:
            continue

        model_acc = get_accuracy(bucket['model_correct'])
        chalk_acc = get_accuracy(bucket['chalk_correct'])

        label = f"{int(low)}-{int(high) if np.isfinite(high) else '∞'}"

        print(
            f"Diff {label:<7} | "
            f"Model: {model_acc:>6}% | "
            f"Chalk: {chalk_acc:>6}% | "
            f"N = {len(bucket)}"
        )

        

    print("\n" + "=" * 65 + "\n")
    for low, high in confidence_bins:

        bucket = df[
            (df['abs_diff'] >= low) &
            (df['abs_diff'] < high)
        ]

        if len(bucket) == 0:
            continue

        model_acc = get_accuracy(bucket['model_correct'])

        label = f"{int(low)}-{int(high) if np.isfinite(high) else '∞'}"

        print(
            f"Diff {label:<7} | "
            f"Model: {model_acc:>6}% | "
            f"N = {len(bucket)}"
        )

        
    print("\n" + "=" * 65 + "\n")
