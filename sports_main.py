"""Train sports betting models for NFL, NBA, MLB, NHL."""
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RANDOM_SEED, RESULTS_DIR
from src.sports_features import build_game_features
from src.sports_fetcher import SPORTS_CONFIG, fetch_historical_games
from src.sports_model import train_and_save_sports, walk_forward_sports


def main():
    np.random.seed(RANDOM_SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary = []
    for league in SPORTS_CONFIG:
        print(f"\n{'='*60}")
        print(f"  {league}")
        print(f"{'='*60}")

        print(f"  Fetching historical games...")
        games = fetch_historical_games(league)
        if games.empty:
            print(f"  No data found for {league}, skipping.")
            continue

        print(f"  {len(games)} games loaded. Building features...")
        feat_df = build_game_features(games)
        feat_df = feat_df.dropna().reset_index(drop=True)
        print(f"  {len(feat_df)} feature rows")

        print(f"  Walk-forward validation (4 folds)...")
        wf = walk_forward_sports(feat_df, model_type="xgboost", n_splits=4)
        print(f"  OOS accuracy: {wf['oos_accuracy']:.4f}")

        print(f"  Training final models on all data...")
        train_and_save_sports(league, feat_df)

        summary.append({"League": league, "Games": len(games), "OOS Accuracy": round(wf["oos_accuracy"], 4)})

    print(f"\n{'='*60}")
    print("SPORTS SUMMARY")
    print(f"{'='*60}")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
