import warnings

warnings.filterwarnings("ignore")

# GENERAL IMPORTS
import os

# this is needed to limit the number of scipy threads
# and let spikeinterface handle parallelization
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import numpy as np
from pathlib import Path
import json
import pickle
import time

# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

from spikeinterface.core.core_tools import SIJsonEncoder

from wavpack_numcodecs import WavPack


data_folder = Path("../data/")
scratch_folder = Path("../scratch/")
results_folder = Path("../results/")


# define argument parser
parser = argparse.ArgumentParser(description="Compress AIND Neurpixels data")

bps_group = parser.add_mutually_exclusive_group()
bps_help = "Wavpack BPS"
bps_group.add_argument("--bps", help=bps_help)
bps_group.add_argument("static_bps", nargs="?", default="", help=bps_help)


if __name__ == "__main__":
    args = parser.parse_args()

    BPS = args.bps or args.static_bps
    if BPS == "":
        BPS = None
    else:
        BPS = float(BPS)

    job_kwargs = {}
    job_kwargs["n_jobs"] = -1
    job_kwargs["progress_bar"] = False
    si.set_global_job_kwargs(**job_kwargs)

    print(f"Running wavpack compression with the following parameters:")
    print(f"\tBPS: {BPS}")

    # load job files
    job_config_files = [p for p in data_folder.iterdir() if (p.suffix == ".json" or p.suffix == ".pickle" or p.suffix == ".pkl") and "job" in p.name]
    print(f"Found {len(job_config_files)} configurations")

    if len(job_config_files) > 0:
        ####### COMPRESSION #######
        print("\n\nCOMPRESSING")
        t_compression_start_all = time.perf_counter()

        for job_config_file in job_config_files:
            t_preprocessing_start = time.perf_counter()
            preprocessing_notes = ""

            if job_config_file.suffix == ".json":
                with open(job_config_file, "r") as f:
                    job_config = json.load(f)
            else:
                with open(job_config_file, "rb") as f:
                    job_config = pickle.load(f)

            session_name = job_config["session_name"]
            recording_name = job_config["recording_name"]
            recording_dict = job_config["recording_dict"]
            skip_times = job_config.get("skip_times", False)

            try:
                recording = si.load_extractor(recording_dict, base_folder=data_folder)
            except:
                raise RuntimeError(
                    f"Could not find load recording {recording_name} from dict. "
                    f"Make sure mapping is correct!"
                )
            if skip_times:
                print("Resetting recording timestamps")
                recording.reset_times()

            compressor = WavPack(bps=BPS)

            print(f"Recording {recording_name}: {recording}")

            recording_compressed = recording.save(
                folder=results_folder / f"{recording_name}.zarr",
                format="zarr", compressor=compressor
            )
            cr = recording_compressed.get_annotation("compression_ratio")
            print(f"Compressed recording. Compression ratio: {cr}")

            job_config["recording_dict"] = recording_compressed.to_dict(
                recursive=True,
                relative_to=results_folder
            )

            with open(results_folder / job_config_file.name, "w") as f:
                json.dump(job_config, f, indent=4, cls=SIJsonEncoder)


