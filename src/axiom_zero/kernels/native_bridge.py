"""Native C++20 shared-library compiler and ctypes bridge with deterministic NumPy fallback."""

from __future__ import annotations

import ctypes
import hashlib
import math
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt

F32Array = npt.NDArray[np.float32]
I32Array = npt.NDArray[np.int32]


def _sample_categorical_py(dist: F32Array, u: float) -> int:
    clamped = min(max(float(u), 0.0), 0.9999999)
    cdf = 0.0
    last_nz = int(dist.shape[0]) - 1
    for idx in range(int(dist.shape[0])):
        p = float(dist[idx])
        if p > 0.0:
            last_nz = idx
            cdf += p
            if clamped < cdf:
                return idx
    return last_nz


class NativeKernelBridge:
    """Compiles and binds `csrc/axiom_kernels.cpp` via C++20 `ctypes`, with exact NumPy fallback."""

    def __init__(self, csrc_path: Path | None = None) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self._src = csrc_path or (repo_root / "csrc" / "axiom_kernels.cpp")
        self._lib: ctypes.CDLL | None = None
        self._backend = "numpy-fallback"
        self._try_compile_and_load()

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def is_native_cpp(self) -> bool:
        return self._lib is not None

    def _try_compile_and_load(self) -> None:
        if not self._src.exists():
            return
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            return

        cache_dir = self._src.parent.parent / ".axiom_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        src_hash = hashlib.sha256(self._src.read_bytes()).hexdigest()[:12]
        ext = "dylib" if platform.system() == "Darwin" else "so"
        out_lib = cache_dir / f"libaxiom_kernels_{src_hash}.{ext}"

        if not out_lib.exists():
            cmd = [
                compiler,
                "-O3",
                "-std=c++20",
                "-shared",
                "-fPIC",
                str(self._src),
                "-o",
                str(out_lib),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                return

        lib = ctypes.CDLL(str(out_lib))
        f_ptr = ctypes.POINTER(ctypes.c_float)
        i_ptr = ctypes.POINTER(ctypes.c_int32)

        lib.axiom_fused_softmax_f32.argtypes = [
            f_ptr,
            f_ptr,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        lib.axiom_fused_softmax_f32.restype = None

        lib.axiom_speculative_verify_step.argtypes = [
            f_ptr,
            f_ptr,
            i_ptr,
            f_ptr,
            ctypes.c_float,
            ctypes.c_int32,
            ctypes.c_int32,
            i_ptr,
        ]
        lib.axiom_speculative_verify_step.restype = ctypes.c_int32

        lib.axiom_paged_kv_attention_f32.argtypes = [
            f_ptr,
            f_ptr,
            f_ptr,
            i_ptr,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            f_ptr,
        ]
        lib.axiom_paged_kv_attention_f32.restype = None

        self._lib = lib
        self._backend = f"c++20-native ({Path(compiler).name} -O3)"

    def fused_softmax(self, logits: F32Array) -> F32Array:
        arr = np.ascontiguousarray(logits, dtype=np.float32)
        if arr.ndim == 1:
            out = np.empty_like(arr)
            if self._lib is not None:
                self._lib.axiom_fused_softmax_f32(
                    arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    1,
                    int(arr.shape[0]),
                )
                return out
            shifted = arr - np.max(arr)
            exps = np.exp(shifted)
            return np.asarray(exps / np.sum(exps), dtype=np.float32)

        rows, cols = int(arr.shape[0]), int(arr.shape[1])
        out = np.empty_like(arr)
        if self._lib is not None:
            self._lib.axiom_fused_softmax_f32(
                arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                rows,
                cols,
            )
            return out
        shifted = arr - np.max(arr, axis=1, keepdims=True)
        exps = np.exp(shifted)
        return np.asarray(exps / np.sum(exps, axis=1, keepdims=True), dtype=np.float32)

    def speculative_verify_step(
        self,
        target_probs: F32Array,
        draft_probs: F32Array,
        draft_tokens: I32Array,
        uniform_r: F32Array,
        resample_u: float,
    ) -> list[int]:
        k_draft = int(draft_tokens.shape[0])
        vocab_size = int(target_probs.shape[1])
        t_contig = np.ascontiguousarray(target_probs, dtype=np.float32)
        d_contig = np.ascontiguousarray(draft_probs, dtype=np.float32)
        tok_contig = np.ascontiguousarray(draft_tokens, dtype=np.int32)
        r_contig = np.ascontiguousarray(uniform_r, dtype=np.float32)

        if self._lib is not None:
            out = np.zeros(k_draft + 1, dtype=np.int32)
            emitted = int(
                self._lib.axiom_speculative_verify_step(
                    t_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    d_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    tok_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    r_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_float(float(resample_u)),
                    k_draft,
                    vocab_size,
                    out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                )
            )
            return [int(x) for x in out[:emitted]]

        accepted: list[int] = []
        for idx in range(k_draft):
            tok = int(tok_contig[idx])
            px = float(t_contig[idx, tok])
            qx = max(float(d_contig[idx, tok]), 1e-12)
            if float(r_contig[idx]) < min(1.0, px / qx):
                accepted.append(tok)
            else:
                diff = np.maximum(0.0, t_contig[idx] - d_contig[idx])
                mass = float(np.sum(diff))
                if mass > 1e-12:
                    res = np.asarray(diff / mass, dtype=np.float32)
                    accepted.append(_sample_categorical_py(res, resample_u))
                else:
                    accepted.append(_sample_categorical_py(t_contig[idx], resample_u))
                return accepted

        accepted.append(_sample_categorical_py(t_contig[k_draft], resample_u))
        return accepted

    def paged_kv_attention(
        self,
        query: F32Array,
        key_blocks: F32Array,
        value_blocks: F32Array,
        block_table: I32Array,
        context_len: int,
    ) -> F32Array:
        head_dim = int(query.shape[0])
        block_size = int(key_blocks.shape[1])
        scale = 1.0 / math.sqrt(head_dim)
        q_c = np.ascontiguousarray(query, dtype=np.float32)
        k_c = np.ascontiguousarray(key_blocks, dtype=np.float32)
        v_c = np.ascontiguousarray(value_blocks, dtype=np.float32)
        bt_c = np.ascontiguousarray(block_table, dtype=np.int32)
        out = np.zeros(head_dim, dtype=np.float32)

        if self._lib is not None:
            self._lib.axiom_paged_kv_attention_f32(
                q_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                k_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                v_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                bt_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                int(context_len),
                block_size,
                head_dim,
                ctypes.c_float(scale),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            return out

        if context_len <= 0:
            return out
        scores = np.empty(context_len, dtype=np.float32)
        vals = np.empty((context_len, head_dim), dtype=np.float32)
        for pos in range(context_len):
            phys = int(bt_c[pos // block_size])
            slot = pos % block_size
            scores[pos] = float(np.dot(q_c, k_c[phys, slot])) * scale
            vals[pos] = v_c[phys, slot]
        weights = self.fused_softmax(scores)
        return np.asarray(weights @ vals, dtype=np.float32)


_GLOBAL_BRIDGE: NativeKernelBridge | None = None


def get_kernel_bridge() -> NativeKernelBridge:
    global _GLOBAL_BRIDGE
    if _GLOBAL_BRIDGE is None:
        _GLOBAL_BRIDGE = NativeKernelBridge()
    return _GLOBAL_BRIDGE
