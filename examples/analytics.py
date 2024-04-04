"""Analytics examples."""

from pathlib import Path

from octopus.analytics.run import OctoAnalitics

folder_path = Path("studies").joinpath("TMM_octofull_all_modules_all_features")

octo_ana = OctoAnalitics(folder_path)

octo_ana.run_analytics()
