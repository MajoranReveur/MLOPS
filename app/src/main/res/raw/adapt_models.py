import argparse
import onnx.hub
import pathlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request

raw_dir = "./"
model_path_stem = "model"

model_path = raw_dir + model_path_stem + ".onnx"
fixed_path = raw_dir + model_path_stem + "_fixed.onnx"

print(model_path)

subprocess.run(
    [
        sys.executable,
        "-m",
        "onnxruntime.tools.make_dynamic_shape_fixed",
        "--dim_param=batch_size",
        "--dim_value=1",
        str(model_path),
        str(fixed_path),
    ],
    check=True,
)