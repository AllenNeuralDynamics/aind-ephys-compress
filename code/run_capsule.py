import warnings

warnings.filterwarnings("ignore")

# GENERAL IMPORTS
import os
import sys

# this is needed to limit the number of scipy threads
# and let spikeinterface handle parallelization
os.environ["OPENBLAS_NUM_THREADS"] = "1"

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
PARAMS_FILE = Path("../params.json")


def run() -> None:
    if not PARAMS_FILE.is_file():
        raise ValueError(f"Params file not found: {PARAMS_FILE}")
    with open(PARAMS_FILE, "r") as f:
        full_params = json.load(f)
    compression_params = full_params["compression"]
    LOGGING = full_params.get("logging", None)

    # setup logging before any other logging call
    if LOGGING is None:
        logging.basicConfig(level="INFO", stream=sys.stdout, format="%(message)s")
    else:
        if LOGGING["package"] == "logging":
            logging_cfg = LOGGING.get("logging_cfg", {})
            logging.basicConfig(stream=sys.stdout, **logging_cfg)
        elif LOGGING["package"] == "log-schema":
            import log_schema

            pipeline_name = LOGGING.get("pipeline_name", "AIND Ephys Hybrid Benchmark")
            acquisition_name = LOGGING.get("acquisition_name", "unknown")

            config = LOGGING.get("logging_cfg")
            if config is not None and len(config) == 0:
                config = None
            log_schema.setup_logging(
                config=config,
                model={
                    "pipeline_name": pipeline_name,
                    "acquisition_name": acquisition_name,
                    "process_name": "Compression"
                }
            )

    logging.info("Begin processing...", extra={"event_type": "stage_start"})

    # COMPRESSION_BPS env variable overrides the params file (one task per compression level)
    BPS = os.getenv("COMPRESSION_BPS") or compression_params.get("bps", None)
    BPS = None if BPS in (None, "") else float(BPS)
    HIGHPASS = compression_params.get("highpass", False)

    # Use CO_CPUS/N_JOBS_EXT env variable if available
    N_JOBS_EXT = os.getenv("CO_CPUS") or os.getenv("N_JOBS_EXT")
    N_JOBS = int(N_JOBS_EXT) if N_JOBS_EXT is not None else -1
    si.set_global_job_kwargs(n_jobs=N_JOBS, progress_bar=False)

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
    logging.info("Pipeline stage completed", extra={"event_type": "stage_complete"})


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        logging.exception("Pipeline stage failed", extra={"event_type": "stage_error"})
        raise
