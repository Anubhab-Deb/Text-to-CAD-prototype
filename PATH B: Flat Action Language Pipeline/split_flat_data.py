import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("flat_dataset/text_action_pairs.csv")
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
train_df.to_csv("train_flat.csv", index=False)
val_df.to_csv("val_flat.csv", index=False)
print(f"Train: {len(train_df)}, Val: {len(val_df)}")
