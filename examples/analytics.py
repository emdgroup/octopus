"""Analytics examples."""

from pathlib import Path

from octopus.analytics.run import OctoAnalitics

folder_path = Path("studies").joinpath("2024_02_27_htm_octofull_chtm_EQE_2")

octo_ana = OctoAnalitics(folder_path)

octo_ana.run_analytics()
