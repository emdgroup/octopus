"""Analytics examples."""

from pathlib import Path

from octopus.analytics.run import OctoAnalitics

# path = (
#     Path("studies")
#     .joinpath("housing_octofull_test")
#     .joinpath("experiment0")
#     .joinpath("sequence0")
#     .joinpath("exp0_0.pkl")
# )

# with open(path, "rb") as f:
#     x = pickle.load(f)

# print(x.predictions["0_0_0"])

folder_path = Path("studies").joinpath("housing_octofull_test_3")

octo_ana = OctoAnalitics(folder_path)

octo_ana.run_analytics()
