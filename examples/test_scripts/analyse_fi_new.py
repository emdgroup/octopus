# analyse experiment

from pathlib import Path

import pandas as pd

from octopus.experiment import OctoExperiment

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

print("Investigate experiment 4")

# path_exp = Path("./studies/20240110B/experiment1/sequence0/exp1_0.pkl")
path_exp = Path("./studies/Titanic/experiment1/sequence0/exp1_0.pkl")
experiment = OctoExperiment.from_pickle(path_exp)

print("Model keys:", experiment.models.keys())

best_bag = experiment.models["best"]

print(best_bag.feature_importances)


############
best_bag.feature_importances.keys()

#############
fi = best_bag.feature_importances["permutation_dev_count"].sort_values(
    by="importance", ascending=False
)
fi = fi[fi["importance"] != 0]
print("Number of nonzero features", len(fi))
fi.head(30)

#############
best_bag.feature_importances["4_0_0"].keys()

#############
fi = best_bag.feature_importances["4_0_0"]["permutation_dev"].sort_values(
    by="importance", ascending=False
)
fi = fi[fi["importance"] != 0]
print("Number of nonzero features", len(fi))
fi.head(30)

########
fi = best_bag.feature_importances["lofo_dev_mean"]
fi.head(50)
