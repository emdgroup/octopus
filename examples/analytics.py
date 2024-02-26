"""Analytics examples."""

from pathlib import Path

from octopus.analytics.run import OctoAnalitics

folder_path = Path("studies").joinpath("housing_octofull_test_3")

octo_ana = OctoAnalitics(folder_path)

octo_ana.run_analytics()
