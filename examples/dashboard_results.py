## Evaluation results via Dashboard

### Necessary imports for this example
from pathlib import Path

from octopus.dashboard.run import OctoDash

### Select path of created study
# In this case we load the data from the basic classification example
folder_path = Path("studies").joinpath("basic_regression_example")

### Start the app
octo_dashboard = OctoDash(folder_path)
octo_dashboard.run()
