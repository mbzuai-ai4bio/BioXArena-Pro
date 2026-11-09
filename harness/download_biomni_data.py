"""Download the Biomni data lake + benchmark files once (about 15 GB) into <root>/biomni_data, exactly as Biomni A1 does on
first use (academic mode, all datasets; our runs had 76 data-lake files and benchmark/{DbQA,hle,SeqQA}).
Usage: python harness/download_biomni_data.py /path/to/biomni_root   then export BIOMNI_DATA_ROOT=/path/to/biomni_root"""
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents" / "Biomni"))
from biomni.env_desc import data_lake_dict  # noqa: E402
from biomni.utils import check_and_download_s3_files  # noqa: E402

root = Path(sys.argv[1]).resolve() / "biomni_data"
lake, bench = root / "data_lake", root / "benchmark"
os.makedirs(lake, exist_ok=True); os.makedirs(bench, exist_ok=True)
check_and_download_s3_files(s3_bucket_url="https://biomni-release.s3.amazonaws.com", local_data_lake_path=str(lake),
                            expected_files=list(data_lake_dict.keys()), folder="data_lake")
if not (bench / "hle").is_dir():
    check_and_download_s3_files(s3_bucket_url="https://biomni-release.s3.amazonaws.com", local_data_lake_path=str(bench),
                                expected_files=[], folder="benchmark")
print(f"data lake files: {len(list(lake.iterdir()))} (expected {len(data_lake_dict)}); benchmark: {sorted(p.name for p in bench.iterdir())}")
