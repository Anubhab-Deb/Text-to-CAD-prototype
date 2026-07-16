import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("text_actions_pairs.csv")
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
train_df.to_csv("text_actions_pairs_train.csv", index=False)
val_df.to_csv("text_actions_pairs_val.csv", index=False)
print(f"Train: {len(train_df)}, Val: {len(val_df)}")
