"""Train sports betting models for NFL, NBA, MLB, NHL."""
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RANDOM_SEED, RESULTS_DIR
from src.sports_features import build_game_features
from src.sports_fetcher import SPORTS_CONFIG, fetch_historical_games
from src.sports_model import train_and_save_sports, walk_forward_sports


def _process_league(league: str) -> dict | None:
    print(f"\n{'='*60}")
    print(f"  {league}")
    print(f"{'='*60}")

    games = fetch_historical_games(league)
    if games.empty:
        print(f"  No data found for {league}, skipping.")
        return None

    print(f"  {len(games)} games loaded. Building features...")
    feat_df = build_game_features(games)
    feat_df = feat_df.dropna().reset_index(drop=True)
    print(f"  {len(feat_df)} feature rows")

    print(f"  Walk-forward validation (3 folds)...")
    wf = walk_forward_sports(feat_df, model_type="xgboost", n_splits=3)
    print(f"  OOS accuracy: {wf['oos_accuracy']:.4f}")

    print(f"  Training final models on all data...")
    train_and_save_sports(league, feat_df)

    return {"League": league, "Games": len(games), "OOS Accuracy": round(wf["oos_accuracy"], 4)}


def main():
    np.random.seed(RANDOM_SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    leagues = list(SPORTS_CONFIG.keys())
    print(f"Training {len(leagues)} leagues in parallel: {', '.join(leagues)}")

    summary = []
    with ThreadPoolExecutor(max_workers=len(leagues)) as pool:
        futures = {pool.submit(_process_league, lg): lg for lg in leagues}
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                summary.append(result)

    print(f"\n{'='*60}")
    print("SPORTS SUMMARY")
    print(f"{'='*60}")
    if summary:
        print(pd.DataFrame(summary).sort_values("League").to_string(index=False))


if __name__ == "__main__":
    main()
