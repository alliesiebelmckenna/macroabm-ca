"""Quick simulation runner for Canada."""
from pathlib import Path

from macro_data import DataWrapper
from macro_data.configuration_utils import default_data_configuration
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation

# ---- CONFIG ----
REPO_ROOT = Path(__file__).resolve().parent.parent
COUNTRY = "CAN"
T_MAX = 20
RAW_DATA = REPO_ROOT / "tests/test_macro_data/unit/sample_raw_data"
OUTPUT = REPO_ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

# ---- DATA PREPROCESSING ----
data_config = default_data_configuration(
    countries=[COUNTRY],
    aggregate_industries=False,
    proxy_country_dict={"CAN": "FRA"},
    use_disagg_can_2014_reader=True,
    seed=0,
)
datawrapper = DataWrapper.from_config(data_config, RAW_DATA, single_hfcs_survey=True)

# ---- SIMULATION ----
n_ind = datawrapper.n_industries
sim_config = SimulationConfiguration(
    country_configurations={COUNTRY: CountryConfiguration.n_industry_default(n_industries=n_ind)},
    t_max=T_MAX,
    seed=0,
)
sim = Simulation.from_datawrapper(datawrapper=datawrapper, simulation_configuration=sim_config)
sim.run()

# ---- SAVE ----
sim.save(save_dir=OUTPUT, file_name=f"sim_{COUNTRY.lower()}.h5")
sim.shallow_hdf_save(save_dir=OUTPUT, file_name=f"sim_{COUNTRY.lower()}_shallow.h5")
print(sim.shallow_df_dict()[COUNTRY].head())