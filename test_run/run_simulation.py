"""Unified simulation runner for any country.

Usage:
    uv run python run_simulation.py                    # France (default)
    uv run python run_simulation.py DEU                # Germany (EU - no proxy needed)
    uv run python run_simulation.py CAN                # Canada (auto-uses FRA as proxy)
    uv run python run_simulation.py CAN --proxy DEU    # Canada with Germany as proxy
    uv run python run_simulation.py USA --proxy FRA    # USA with France as proxy
    uv run python run_simulation.py GBR --proxy DEU    # UK with Germany as proxy
    uv run python run_simulation.py CAN --scale 5000   # Custom scale
"""
import argparse
import sys
import time
from pathlib import Path

from macro_data import DataWrapper
from macro_data.configuration_utils import default_data_configuration
from macromodel.configurations import CountryConfiguration, SimulationConfiguration
from macromodel.simulation import Simulation


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def main():
    total_start = time.perf_counter()
    
    parser = argparse.ArgumentParser(description="Run macroeconomic simulation for a country.")
    parser.add_argument("country", nargs="?", default="FRA", help="Country code (default: FRA)")
    parser.add_argument("--proxy", default=None, help="Proxy country for non-EU countries (default: FRA for non-EU)")
    parser.add_argument("--scale", type=int, default=10_000, help="Scale factor (default: 10000)")
    parser.add_argument("--t-max", type=int, default=20, help="Number of timesteps (default: 20)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--output", default="output", help="Output directory (default: output)")
    args = parser.parse_args()

    country = args.country.upper()
    proxy = args.proxy.upper() if args.proxy else None

    # Resolve paths
    repo_root = Path(__file__).resolve().parent.parent
    raw_data = repo_root / "tests/test_macro_data/unit/sample_raw_data"
    output_dir = repo_root / args.output
    output_dir.mkdir(exist_ok=True)

    # Determine if country needs proxy (non-EU countries)
    # EU28 countries as of 2014 (includes UK which was still EU member)
    eu_countries = {
        "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
        "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
        "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE", "GBR"
    }
    
    needs_proxy = country not in eu_countries
    proxy_country = proxy or ("FRA" if needs_proxy else None)

    print(f"=== Running simulation for {country} ===")
    print(f"  Scale: {args.scale:,}")
    print(f"  Timesteps: {args.t_max}")
    print(f"  Seed: {args.seed}")
    
    if needs_proxy:
        print(f"  Proxy country: {proxy_country} (required for non-EU)")
    else:
        print(f"  Proxy country: {proxy_country or 'None (EU country)'}")

    # Build proxy dict if needed
    proxy_dict = {country: proxy_country} if proxy_country and needs_proxy else {}

    # Special handling for Canada
    is_canada = country == "CAN"
    use_disagg_can = is_canada and proxy_country in eu_countries

    if is_canada and not use_disagg_can:
        print("  Note: Using standard Canada configuration")
    elif use_disagg_can:
        print("  Note: Using disaggregated Canada energy sector reader")

    # 1. Data preprocessing
    print("\n[1/4] Preprocessing data...")
    step_start = time.perf_counter()
    data_config = default_data_configuration(
        countries=[country],
        proxy_country_dict=proxy_dict if proxy_dict else None,
        scale={country: args.scale},
        seed=args.seed,
        use_disagg_can_2014_reader=use_disagg_can,
        aggregate_industries=not use_disagg_can,  # Canada disagg needs non-aggregated
    )
    datawrapper = DataWrapper.from_config(data_config, raw_data, single_hfcs_survey=True)
    step_elapsed = time.perf_counter() - step_start
    print(f"  Created {datawrapper.n_industries} industries in {format_duration(step_elapsed)}")

    # 2. Simulation configuration
    step_start = time.perf_counter()
    print("\n[2/4] Configuring simulation...")
    n_industries = datawrapper.n_industries
    
    if use_disagg_can:
        # Canada needs special industry configuration
        country_config = CountryConfiguration.n_industry_default(n_industries=n_industries)
    else:
        country_config = CountryConfiguration()

    sim_config = SimulationConfiguration(
        country_configurations={country: country_config},
        t_max=args.t_max,
        seed=args.seed,
    )
    step_elapsed = time.perf_counter() - step_start
    print(f"  Configuration complete in {format_duration(step_elapsed)}")

    # 3. Run simulation
    print("\n[3/4] Running simulation...")
    step_start = time.perf_counter()
    simulation = Simulation.from_datawrapper(
        datawrapper=datawrapper, 
        simulation_configuration=sim_config
    )
    simulation.run()
    step_elapsed = time.perf_counter() - step_start
    print(f"  Simulation complete in {format_duration(step_elapsed)}")
    print(f"  Average per timestep: {step_elapsed / args.t_max:.2f}s")

    # 4. Save results
    print("\n[4/4] Saving results...")
    sim_file = output_dir / f"sim_{country.lower()}.h5"
    shallow_file = output_dir / f"sim_{country.lower()}_shallow.h5"
    
    simulation.save(save_dir=output_dir, file_name=sim_file.name)
    simulation.shallow_hdf_save(save_dir=output_dir, file_name=shallow_file.name)
    
    print(f"  Full output: {sim_file}")
    print(f"  Shallow output: {shallow_file}")

    # Print summary
    results = simulation.shallow_df_dict()
    print(f"\n=== Summary for {country} ===")
    print(results[country].head())

    # Final timing
    total_elapsed = time.perf_counter() - total_start
    print(f"\n=== Timing Summary ===")
    print(f"  Total elapsed: {format_duration(total_elapsed)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
