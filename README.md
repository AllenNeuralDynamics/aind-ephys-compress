# aind-capsule-template

Basic structure of a capsule to be customized as needed.

## Parameters

Parameters are read from the `compression` section of `../params.json` (the capsule fails if the file is missing):

| Key | Default | Description |
|-----|---------|-------------|
| `bps` | `null` | WavPack bits per sample. `null` or `""` means lossless |
| `highpass` | `false` | Whether to highpass the recording prior to compression |

If the `COMPRESSION_BPS` environment variable is set and non-empty, it overrides `compression.bps`.

Minimal example `params.json`:

```json
{
    "compression": {
        "bps": null,
        "highpass": false
    }
}
```

The number of jobs can be set with the `CO_CPUS` or `N_JOBS_EXT` environment variables (default: all cores).
