import warnings

warnings.filterwarnings("ignore")

# GENERAL IMPORTS
import os
import sys

# this is needed to limit the number of scipy threads
# and let spikeinterface handle parallelization
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import numpy as np
from pathlib import Path
import json
import pickle
import time
import logging

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

highpass_group = parser.add_mutually_exclusive_group()
highpass_help = "Whether to highpass the recording prior to compression"
highpass_group.add_argument("--highpass", help=highpass_help)
highpass_group.add_argument("static_highpass", nargs="?", default="false", help=highpass_help)


if __name__ == "__main__":
    args = parser.parse_args()

    BPS = args.bps or args.static_bps
    if BPS == "":
        BPS = None
    else:
        BPS = float(BPS)
    HIGHPASS = (
        args.static_highpass.lower() == "true" if args.static_highpass
        else args.highpass
    )

    # Use CO_CPUS/SLURM_CPUS_ON_NODE env variable if available
    N_JOBS_EXT = os.getenv("CO_CPUS") or os.getenv("SLURM_CPUS_ON_NODE")
    N_JOBS = int(N_JOBS_EXT) if N_JOBS_EXT is not None else -1
    si.set_global_job_kwargs(n_jobs=N_JOBS, progress_bar=False)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")

    bps_config_files = [
        p for p in data_folder.iterdir() if p.name.endswith(".txt") and "bps" in p.name
    ]
    if len(bps_config_files) == 1:
        bps_config_file = bps_config_files[0]
        logging.info(f"Loading BPS from {bps_config_file}")
        BPS = float(bps_config_file.read_text())

    logging.info(f"Running wavpack compression with the following parameters:")
    logging.info(f"\tBPS: {BPS}")
    logging.info(f"\tHIGHPASS: {HIGHPASS}")

    # load job files
    job_config_files = [p for p in data_folder.iterdir() if (p.suffix == ".json" or p.suffix == ".pickle" or p.suffix == ".pkl") and "job" in p.name]
    logging.info(f"Found {len(job_config_files)} configurations")

    if len(job_config_files) > 0:
        ####### COMPRESSION #######
        logging.info("\n\nCOMPRESSING")
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
                recording = si.load(recording_dict, base_folder=data_folder)
            except:
                raise RuntimeError(
                    f"Could not find load recording {recording_name} from dict. "
                    f"Make sure mapping is correct!"
                )
            if skip_times:
                logging.info("Resetting recording timestamps")
                recording.reset_times()

            if HIGHPASS:
                logging.info("Applying highpass filter")
                recording = spre.highpass_filter(recording)

            compressor = WavPack(bps=BPS)

            logging.info(f"Recording {recording_name}: {recording}")

            recording_compressed = recording.save(
                folder=results_folder / f"{recording_name}.zarr",
                format="zarr", compressor=compressor
            )
            cr = recording_compressed.get_annotation("compression_ratio")
            logging.info(f"Compressed recording. Compression ratio: {cr}")

            job_config["recording_dict"] = recording_compressed.to_dict(
                recursive=True,
                relative_to=results_folder
            )

            with open(results_folder / f"{job_config_file.stem}.json", "w") as f:
                json.dump(job_config, f, indent=4, cls=SIJsonEncoder)


