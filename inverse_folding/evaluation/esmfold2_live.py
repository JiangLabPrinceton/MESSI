"""Persistent ESMFold2 inference isolated from fair-esm package collisions.

The Fusion process and this worker both run with the current Python executable
(``immune-design`` on the cluster). The worker imports torch first, then overlays
Biohub's ``esm`` and ``transformers`` packages from an explicitly supplied
site-packages directory. Keeping that overlay in a subprocess prevents Biohub's
top-level ``esm`` package from replacing fair-esm in the Fusion process.

The model is loaded once. Fold requests are exchanged as JSON lines and written
to the shared refold cache as ``<key>.pdb`` plus ``<key>.plddt``.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

READY_PREFIX = "ESMFOLD2_WORKER_READY"
RESULT_PREFIX = "ESMFOLD2_WORKER_RESULT"


def _validated_site_packages(site_packages: str) -> Path:
    site = Path(site_packages).expanduser().resolve()
    required = (
        site / "esm" / "models" / "esmfold2",
        site / "transformers" / "models" / "esmfold2",
    )
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "ESMFold2 Biohub package overlay is incomplete; missing " + ", ".join(missing)
        )
    return site


def build_worker_command(
    *,
    site_packages: str,
    python_executable: str | None = None,
    device: str = "cuda",
    model_name: str = "biohub/ESMFold2",
    num_loops: int = 3,
    num_sampling_steps: int = 50,
    num_diffusion_samples: int = 1,
    seed: int = 0,
) -> list[str]:
    """Build the worker argv; no shell or conda activation is involved."""
    site = _validated_site_packages(site_packages)
    return [
        python_executable or sys.executable,
        "-u",
        "-m",
        "inverse_folding.evaluation.esmfold2_live",
        "worker",
        "--site-packages",
        str(site),
        "--device",
        device,
        "--model-name",
        model_name,
        "--num-loops",
        str(num_loops),
        "--num-sampling-steps",
        str(num_sampling_steps),
        "--num-diffusion-samples",
        str(num_diffusion_samples),
        "--seed",
        str(seed),
    ]


class ESMFold2LiveClient:
    """One persistent ESMFold2 worker used by repeated Fusion refolds."""

    def __init__(
        self,
        *,
        site_packages: str,
        device: str = "cuda",
        model_name: str = "biohub/ESMFold2",
        num_loops: int = 3,
        num_sampling_steps: int = 50,
        num_diffusion_samples: int = 1,
        seed: int = 0,
        python_executable: str | None = None,
        popen_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        command = build_worker_command(
            site_packages=site_packages,
            python_executable=python_executable,
            device=device,
            model_name=model_name,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=num_diffusion_samples,
            seed=seed,
        )
        worker_env = os.environ.copy()
        # The immune-design conda activation may export INTEL while Biohub's
        # compiled dependencies load GNU OpenMP. A fresh worker otherwise exits
        # in mkl-service before it can report a protocol error.
        worker_env["MKL_THREADING_LAYER"] = "GNU"
        self._process = popen_factory(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=worker_env,
        )
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("ESMFold2 worker did not expose stdin/stdout pipes")
        self._next_request_id = 1
        self._closed = False
        self.metadata = self._read_prefixed(READY_PREFIX)
        if self.metadata.get("error"):
            self.close()
            raise RuntimeError(
                "ESMFold2 worker failed during model load: "
                f"{self.metadata.get('error_type', 'RuntimeError')}: {self.metadata['error']}"
            )
        atexit.register(self.close)

    def _read_prefixed(self, prefix: str) -> dict[str, Any]:
        marker = prefix + "\t"
        while True:
            line = self._process.stdout.readline()
            if line == "":
                returncode = self._process.poll()
                raise RuntimeError(
                    f"ESMFold2 worker exited before {prefix} (returncode={returncode})"
                )
            if line.startswith(marker):
                payload = line[len(marker):].strip()
                try:
                    value = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid ESMFold2 worker response: {payload!r}") from exc
                if not isinstance(value, dict):
                    raise RuntimeError("ESMFold2 worker response must be a JSON object")
                return value

    def predict(
        self,
        *,
        sequence: str,
        protein_id: str,
        design_id: str,
        cache_dir: str | None,
        msa_a3m_path: str | None = None,
    ) -> dict[str, Any]:
        """Fold a single chain. Pass ``msa_a3m_path`` (an a3m) to condition on an MSA — optional
        and off by default, so existing MSA-free callers are unchanged. MSA-conditioned results are
        cached under a distinct key (``_cache_key_with_msa``)."""
        if self._closed:
            raise RuntimeError("ESMFold2 live model is closed")
        if cache_dir is None:
            raise RuntimeError("ESMFold2 live backend requires cache_dir")

        cached = _cached_prediction(protein_id, sequence, cache_dir, msa_a3m_path)
        if cached is not None:
            return cached

        request_id = self._next_request_id
        self._next_request_id += 1
        request = {
            "op": "fold",
            "request_id": request_id,
            "sequence": sequence,
            "protein_id": protein_id,
            "design_id": design_id,
            "cache_dir": cache_dir,
            "msa_a3m_path": msa_a3m_path,
        }
        self._process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self._process.stdin.flush()
        response = self._read_prefixed(RESULT_PREFIX)
        if response.get("request_id") != request_id:
            raise RuntimeError(
                "ESMFold2 worker response id mismatch: "
                f"expected {request_id}, got {response.get('request_id')!r}"
            )
        if response.get("error"):
            raise RuntimeError(
                "ESMFold2 fold failed: "
                f"{response.get('error_type', 'RuntimeError')}: {response['error']}"
            )
        return response

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        try:
            process.stdin.write('{"op":"shutdown"}\n')
            process.stdin.flush()
            process.wait(timeout=10)
        except Exception:  # noqa: BLE001 - cleanup must not mask the caller's error
            process.terminate()


def _cache_key_with_msa(protein_id: str, sequence: str, msa_a3m_path: str | None) -> str:
    """cache_key(protein_id, sequence), suffixed with an MSA fingerprint when an MSA is used so
    MSA-conditioned folds never collide with the shared MSA-free cache (reused across backends)."""
    from inverse_folding.evaluation.esmfold_runner import cache_key

    key = cache_key(protein_id, sequence)
    if msa_a3m_path:
        import hashlib

        key = f"{key}.msa{hashlib.sha1(str(msa_a3m_path).encode()).hexdigest()[:8]}"
    return key


def _cached_prediction(
    protein_id: str, sequence: str, cache_dir: str, msa_a3m_path: str | None = None
) -> dict[str, Any] | None:
    key = _cache_key_with_msa(protein_id, sequence, msa_a3m_path)
    pdb_path = os.path.join(cache_dir, f"{key}.pdb")
    plddt_path = os.path.join(cache_dir, f"{key}.plddt")
    if not (os.path.isfile(pdb_path) and os.path.isfile(plddt_path)):
        return None
    with open(plddt_path) as handle:
        plddt = float(handle.read().strip())
    return {
        "pdb_string": None,
        "pLDDT": plddt,
        "pdb_path": pdb_path,
        "cache_hit": True,
    }


def _module_is_below(module: Any, root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    return module_file is not None and Path(module_file).resolve().is_relative_to(root)


def _load_worker_runtime(args: argparse.Namespace) -> dict[str, Any]:
    # Import torch before adding the overlay. This deliberately pins the worker
    # to immune-design's torch build instead of the dedicated env's torch wheel.
    import torch

    site = _validated_site_packages(args.site_packages)
    if "esm" in sys.modules or "transformers" in sys.modules:
        raise RuntimeError("worker imported esm/transformers before installing the Biohub overlay")
    sys.path.insert(0, str(site))

    import esm
    import transformers
    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        ProteinInput,
        StructurePredictionInput,
        MSA,
    )
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    if not _module_is_below(esm, site) or not _module_is_below(transformers, site):
        raise RuntimeError(
            "Biohub overlay was not selected for both esm and transformers; "
            f"esm={getattr(esm, '__file__', None)}, "
            f"transformers={getattr(transformers, '__file__', None)}"
        )

    model = ESMFold2Model.from_pretrained(args.model_name).to(args.device).eval()
    return {
        "torch": torch,
        "model": model,
        "builder": ESMFold2InputBuilder(),
        "ProteinInput": ProteinInput,
        "StructurePredictionInput": StructurePredictionInput,
        "MSA": MSA,
        "metadata": {
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "torch_module": str(Path(torch.__file__).resolve()),
            "esm_module": str(Path(esm.__file__).resolve()),
            "transformers_module": str(Path(transformers.__file__).resolve()),
            "transformers_version": getattr(transformers, "__version__", None),
            "model_name": args.model_name,
            "device": args.device,
        },
    }


def _fold_request(
    runtime: dict[str, Any], request: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    from inverse_folding.evaluation.refold_normalize import normalize_to_cache

    sequence = str(request["sequence"])
    protein_id = str(request["protein_id"])
    cache_dir = str(request["cache_dir"])
    msa_a3m_path = request.get("msa_a3m_path") or None
    if not sequence:
        raise ValueError("sequence must be non-empty")

    cached = _cached_prediction(protein_id, sequence, cache_dir, msa_a3m_path)
    if cached is not None:
        return cached

    msa = runtime["MSA"].from_a3m(msa_a3m_path) if msa_a3m_path else None
    structure_input = runtime["StructurePredictionInput"](
        sequences=[runtime["ProteinInput"](id="A", sequence=sequence, msa=msa)]
    )
    with runtime["torch"].inference_mode():
        result = runtime["builder"].fold(
            runtime["model"],
            structure_input,
            num_loops=args.num_loops,
            num_sampling_steps=args.num_sampling_steps,
            num_diffusion_samples=args.num_diffusion_samples,
            seed=args.seed,
        )

    mean_plddt = float(result.plddt.mean())
    ptm = float(result.ptm)
    cif_text = result.complex.to_mmcif()
    key = _cache_key_with_msa(protein_id, sequence, msa_a3m_path)
    os.makedirs(cache_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cif", prefix=f".{key}.", dir=cache_dir, delete=False
    ) as handle:
        handle.write(cif_text)
        cif_path = handle.name
    try:
        paths = normalize_to_cache(
            cache_dir,
            key,
            cif_path=cif_path,
            mean_plddt=mean_plddt,
            native_scale="0-1",
        )
    finally:
        Path(cif_path).unlink(missing_ok=True)

    return {
        "pdb_string": None,
        "pLDDT": mean_plddt * 100.0,
        "pTM": ptm,
        "pdb_path": paths["pdb_path"],
        "cache_hit": False,
    }


def _emit(prefix: str, payload: dict[str, Any]) -> None:
    sys.stdout.write(prefix + "\t" + json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _run_worker(args: argparse.Namespace) -> int:
    try:
        with contextlib.redirect_stdout(sys.stderr):
            runtime = _load_worker_runtime(args)
    except Exception as exc:  # noqa: BLE001 - report startup failure over the protocol
        traceback.print_exc(file=sys.stderr)
        _emit(READY_PREFIX, {"error_type": type(exc).__name__, "error": str(exc)})
        return 1

    _emit(READY_PREFIX, runtime["metadata"])
    for line in sys.stdin:
        request: Any = None
        try:
            request = json.loads(line)
            if request.get("op") == "shutdown":
                return 0
            if request.get("op") != "fold":
                raise ValueError(f"unsupported worker operation: {request.get('op')!r}")
            with contextlib.redirect_stdout(sys.stderr):
                response = _fold_request(runtime, request, args)
            response["request_id"] = request.get("request_id")
        except Exception as exc:  # noqa: BLE001 - one failed candidate must not kill the model
            traceback.print_exc(file=sys.stderr)
            response = {
                "request_id": request.get("request_id") if isinstance(request, dict) else None,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        _emit(RESULT_PREFIX, response)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--site-packages", required=True)
    worker.add_argument("--device", default="cuda")
    worker.add_argument("--model-name", default="biohub/ESMFold2")
    worker.add_argument("--num-loops", type=int, default=3)
    worker.add_argument("--num-sampling-steps", type=int, default=50)
    worker.add_argument("--num-diffusion-samples", type=int, default=1)
    worker.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "worker":
        return _run_worker(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
