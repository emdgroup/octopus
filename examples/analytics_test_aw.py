"""Analytics examples."""

from pathlib import Path

from octopus.analytics.run import OctoAnalitics

folder_path = Path("studies").joinpath(
    "20240223A_Martin_wf2_octofull_7x6_poly_global_ridge"
)

octo_ana = OctoAnalitics(folder_path)

octo_ana.run_analytics()
