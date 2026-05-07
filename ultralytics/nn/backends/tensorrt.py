# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import os
from collections import OrderedDict, namedtuple
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import IS_JETSON, LOGGER, PYTHON_VERSION
from ultralytics.utils.checks import check_requirements, check_tensorrt, check_version

from .base import BaseBackend


class TensorRTBackend(BaseBackend):
    """NVIDIA TensorRT inference backend for GPU-accelerated deployment.

    Loads and runs inference with NVIDIA TensorRT serialized engines (.engine files). Supports both TensorRT 7-9 and
    TensorRT 10+ APIs, dynamic input shapes, FP16 precision, and DLA core offloading.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load an NVIDIA TensorRT engine from a serialized .engine file.

        Args:
            weight (str | Path): Path to the .engine file with optional embedded metadata.
        """
        LOGGER.info(f"Loading {weight} for TensorRT inference...")

        if IS_JETSON and check_version(PYTHON_VERSION, "<=3.8.10"):
            check_requirements("numpy==1.23.5")

        try:
            import tensorrt as trt
        except ImportError:
            check_tensorrt()
            import tensorrt as trt

        check_version(trt.__version__, ">=7.0.0", hard=True)
        check_version(trt.__version__, "!=10.2.0", msg="https://github.com/ultralytics/ultralytics/pull/24367")

        if self.device.type == "cpu":
            self.device = torch.device("cuda:0")

        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        logger = trt.Logger(trt.Logger.INFO)

        # Read engine file
        with open(weight, "rb") as f, trt.Runtime(logger) as runtime:
            try:
                meta_len = int.from_bytes(f.read(4), byteorder="little")
                metadata = json.loads(f.read(meta_len).decode("utf-8"))
                dla = metadata.get("dla", None)
                if dla is not None:
                    runtime.DLA_core = int(dla)
            except UnicodeDecodeError:
                f.seek(0)
                metadata = None
            engine = runtime.deserialize_cuda_engine(f.read())
            self.apply_metadata(metadata)
        try:
            self.context = engine.create_execution_context()
        except Exception as e:
            LOGGER.error("TensorRT model exported with a different version than expected\n")
            raise e

        # Setup bindings
        self.bindings = OrderedDict()
        self.output_names = []
        self.fp16 = False
        self.dynamic = False
        self.is_trt10 = not hasattr(engine, "num_bindings")
        num = range(engine.num_io_tensors) if self.is_trt10 else range(engine.num_bindings)

        for i in num:
            if self.is_trt10:
                name = engine.get_tensor_name(i)
                dtype = trt.nptype(engine.get_tensor_dtype(name))
                is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                shape = tuple(engine.get_tensor_shape(name))
                profile_shape = tuple(engine.get_tensor_profile_shape(name, 0)[2]) if is_input else None
            else:
                name = engine.get_binding_name(i)
                dtype = trt.nptype(engine.get_binding_dtype(i))
                is_input = engine.binding_is_input(i)
                shape = tuple(engine.get_binding_shape(i))
                profile_shape = tuple(engine.get_profile_shape(0, i)[1]) if is_input else None

            if is_input:
                if -1 in shape:
                    self.dynamic = True
                    if self.is_trt10:
                        self.context.set_input_shape(name, profile_shape)
                    else:
                        self.context.set_binding_shape(i, profile_shape)
                if dtype == np.float16:
                    self.fp16 = True
            else:
                self.output_names.append(name)

            shape = (
                tuple(self.context.get_tensor_shape(name))
                if self.is_trt10
                else tuple(self.context.get_binding_shape(i))
            )
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))

        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.model = engine

        self._graph = None
        self._graph_stream = None
        self._graph_outputs = None
        self._graph_event = None
        self._post_stream = None
        self._clone_sets = None
        self._clone_idx = 0
        self._consumer_done_event = None
        self._pipeline_post = os.environ.get("ULTRALYTICS_TRT_PIPELINE_POST", "0") == "1"
        self._use_graph = (
            not self.dynamic and self.is_trt10 and os.environ.get("ULTRALYTICS_TRT_CUDA_GRAPH", "1") != "0"
        )
        if self._use_graph:
            for name, binding in self.bindings.items():
                self.context.set_tensor_address(name, binding.ptr)

    @property
    def input_tensor(self) -> torch.Tensor | None:
        """Return the fixed TRT input binding as a torch tensor, or None if not a static engine."""
        if getattr(self, "_use_graph", False):
            return self.bindings["images"].data
        return None

    @property
    def post_stream(self) -> torch.cuda.Stream | None:
        """Dedicated CUDA stream postprocess should run on when pipelining is enabled."""
        return self._post_stream if getattr(self, "_pipeline_post", False) else None

    @property
    def produce_event(self) -> torch.cuda.Event | None:
        """Event signaling that clone of TRT outputs is complete on graph_stream."""
        return self._graph_event if getattr(self, "_pipeline_post", False) else None

    @property
    def last_clone_slot(self) -> int | None:
        """Ring-buffer slot index of the most recent `forward()` output. Use with `mark_slot_consumed`."""
        return getattr(self, "_last_issued_slot", None)

    def mark_slot_consumed(self, slot: int, stream: torch.cuda.Stream) -> None:
        """Record that `stream` has finished reading the clone at `slot`.

        Next replay that reuses `slot` will wait on this event, preventing
        the clone-copy from overwriting in-flight reads.
        """
        if self._pipeline_post and self._slot_consumed is not None:
            self._slot_consumed[slot].record(stream)

    def forward(self, im: torch.Tensor) -> list[torch.Tensor]:
        """Run NVIDIA TensorRT inference with dynamic shape handling.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format on the CUDA device.

        Returns:
            (list[torch.Tensor]): Model predictions as a list of tensors on the CUDA device.
        """
        if self.dynamic and im.shape != self.bindings["images"].shape:
            if self.is_trt10:
                self.context.set_input_shape("images", im.shape)
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name in self.output_names:
                    self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(name)))
            else:
                i = self.model.get_binding_index("images")
                self.context.set_binding_shape(i, im.shape)
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))

        s = self.bindings["images"].shape
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"

        if self._use_graph:
            input_buf = self.bindings["images"].data
            if self._graph is None:
                if im.data_ptr() != input_buf.data_ptr():
                    input_buf.copy_(im)
                self._graph_stream = torch.cuda.Stream(device=self.device)
                self._graph_stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(self._graph_stream):
                    for _ in range(3):
                        self.context.execute_async_v3(self._graph_stream.cuda_stream)
                self._graph_stream.synchronize()
                self._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._graph, stream=self._graph_stream):
                    self.context.execute_async_v3(self._graph_stream.cuda_stream)
                self._graph_outputs = [self.bindings[x].data for x in sorted(self.output_names)]
                self._graph_event = torch.cuda.Event()
                if self._pipeline_post:
                    self._post_stream = torch.cuda.Stream(device=self.device)
                    self._consumer_done_event = torch.cuda.Event()
                    # Ring of clone sets; needs to be large enough that the next wrap
                    # doesn't overwrite a slot whose consumer (postprocess) hasn't
                    # finished reading. With depth-2 pipelining and async postprocess
                    # that doesn't sync, decode may trail by several frames.
                    ring_size = int(os.environ.get("ULTRALYTICS_TRT_CLONE_RING", "8"))
                    sorted_names = sorted(self.output_names)
                    self._clone_sets = [
                        [torch.empty_like(self.bindings[n].data) for n in sorted_names] for _ in range(ring_size)
                    ]
                    # Per-slot "consumer done reading" events. Postprocess calls
                    # `mark_slot_consumed(idx)` after it has read the clone; the
                    # next replay that will reuse this slot waits on this event.
                    self._slot_consumed = [torch.cuda.Event() for _ in range(ring_size)]
                    self._last_issued_slot = None
            elif im.data_ptr() != input_buf.data_ptr():
                input_buf.copy_(im, non_blocking=True)
            current = torch.cuda.current_stream(self.device)
            self._graph_stream.wait_stream(current)
            if self._pipeline_post and self._consumer_done_event is not None:
                # Clone ring slot about to be used: wait for its previous consumer
                # to finish reading it. On the first wrap each slot's event is
                # un-recorded (no-op wait).
                self._graph_stream.wait_event(self._slot_consumed[self._clone_idx])
            # CUDAGraph.replay() launches on the current stream, so switch to
            # graph_stream for the launch and all follow-up clone work.
            with torch.cuda.stream(self._graph_stream):
                self._graph.replay()
                if self._pipeline_post:
                    sorted_names = sorted(self.output_names)
                    idx = self._clone_idx
                    self._clone_idx = (idx + 1) % len(self._clone_sets)
                    clones = self._clone_sets[idx]
                    for c, name in zip(clones, sorted_names):
                        c.copy_(self.bindings[name].data, non_blocking=True)
                    self._graph_event.record(self._graph_stream)
                    self._consumer_done_event.record(self._graph_stream)
                    self._last_issued_slot = idx
            if self._pipeline_post:
                current.wait_stream(self._graph_stream)
                return list(clones)
            current.wait_stream(self._graph_stream)
            return self._graph_outputs

        self.binding_addrs["images"] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        return [self.bindings[x].data for x in sorted(self.output_names)]
