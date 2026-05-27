from pathlib import Path

from macro_data import DataWrapper
from macro_data.configuration_utils import default_data_configuration
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation

# 1. Preprocess data (same pattern as the tests)
data_config = default_data_configuration(countries=["FRA"], seed=0)
# Resolve paths relative to the repo root (where this script lives inside test_run/)
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = REPO_ROOT / "tests/test_macro_data/unit/sample_raw_data"
OUTPUT = REPO_ROOT / "output"
datawrapper = DataWrapper.from_config(data_config, RAW_DATA, single_hfcs_survey=True)

# 2. Configure and run simulation
sim_config = SimulationConfiguration(
    country_configurations={"FRA": CountryConfiguration()},
    t_max=20  # 20 timesteps
)
simulation = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=sim_config)
simulation.run()

# 3. Get results
results = simulation.shallow_df_dict()
print(results["FRA"].head())

# 4. Save
OUTPUT.mkdir(exist_ok=True)
simulation.save(save_dir=OUTPUT, file_name="sim_fra.h5")