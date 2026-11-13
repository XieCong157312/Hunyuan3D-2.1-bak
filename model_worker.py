"""Hunyuan3D-2.1 worker with strictly sequential Shape/Paint GPU residency.

Shape and Paint are instantiated on demand. The Shape pipeline is destroyed and
CUDA cache cleanup is performed before the Paint pipeline is instantiated.
"""
import base64
import gc
import os
import sys
import threading
import time
import uuid
from io import BytesIO
from typing import Callable, Optional

from PIL import Image
import torch

sys.path.insert(0, "./hy3dshape")
sys.path.insert(0, "./hy3dpaint")

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found; continuing without compatibility fix")
except Exception as exc:
    print(f"Warning: torchvision compatibility fix failed: {exc}")

from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.utils import logger
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dpaint.convert_utils import create_glb_with_pbr_materials


def release_cuda_memory():
    """Release unreferenced CUDA allocations before the next model is loaded."""
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as exc:
            logger.warning("CUDA cache cleanup warning: %s", exc)


def quick_convert_with_obj2gltf(obj_path: str, glb_path: str):
    """Convert Paint's OBJ + PBR texture files into a GLB."""
    textures = {
        "albedo": obj_path.replace(".obj", ".jpg"),
        "metallic": obj_path.replace(".obj", "_metallic.jpg"),
        "roughness": obj_path.replace(".obj", "_roughness.jpg"),
    }
    create_glb_with_pbr_materials(obj_path, textures, glb_path)


def load_image_from_base64(encoded_image: str) -> Image.Image:
    """Decode a base64 image and detach it from the BytesIO lifetime."""
    if not encoded_image:
        raise ValueError("No input image provided")
    # Accept both plain base64 and data URLs (data:image/png;base64,...).
    if "," in encoded_image and encoded_image.lstrip().startswith("data:"):
        encoded_image = encoded_image.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded_image, validate=True)
        with Image.open(BytesIO(raw)) as image:
            image.load()
            return image.copy()
    except Exception as exc:
        raise ValueError(f"Invalid base64 image: {exc}") from exc


class ModelWorker:
    """A single-GPU worker. Only one pipeline is resident at a time."""

    def __init__(
        self,
        model_path="tencent/Hunyuan3D-2.1",
        subfolder="hunyuan3d-dit-v2-1",
        device="cuda",
        low_vram_mode=True,
        worker_id=None,
        model_semaphore=None,
        save_dir="gradio_cache",
        mc_algo="mc",
        enable_flashvdm=False,
        compile=False,
        paint_max_num_view=4,
        paint_resolution=512,
    ):
        self.model_path = model_path
        self.subfolder = subfolder  # Kept for compatibility with existing launch scripts.
        self.worker_id = worker_id or str(uuid.uuid4())[:6]
        self.device = device
        self.low_vram_mode = low_vram_mode
        self.model_semaphore = model_semaphore
        self.save_dir = save_dir
        self.mc_algo = mc_algo
        self.enable_flashvdm = enable_flashvdm
        self.compile = compile
        self.paint_max_num_view = paint_max_num_view
        self.paint_resolution = paint_resolution

        self.pipeline = None
        self.paint_pipeline = None
        # This lock covers both stages and teardown; no two requests can overlap on the GPU.
        self._gpu_lock = threading.RLock()
        os.makedirs(self.save_dir, exist_ok=True)

        self.rembg = BackgroundRemover()
        logger.info(
            "Worker %s initialized; Shape/Paint pipelines will be loaded on demand",
            self.worker_id,
        )

    def _load_shape(self):
        if self.pipeline is None:
            logger.info("Loading Shape pipeline")
            pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(self.model_path)
            if self.enable_flashvdm:
                algo = "mc" if self.device in ("cpu", "mps") else self.mc_algo
                pipe.enable_flashvdm(mc_algo=algo)
            if self.compile:
                pipe.compile()
            self.pipeline = pipe
        return self.pipeline

    def _load_paint(self):
        if self.paint_pipeline is None:
            logger.info(
                "Loading Paint pipeline (max_num_view=%s, resolution=%s)",
                self.paint_max_num_view,
                self.paint_resolution,
            )
            # These are the constructor's actual parameter names in Hunyuan3D-2.1.
            conf = Hunyuan3DPaintConfig(
                max_num_view=self.paint_max_num_view,
                resolution=self.paint_resolution,
            )
            conf.device = self.device
            conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
            conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
            conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
            self.paint_pipeline = Hunyuan3DPaintPipeline(conf)
        return self.paint_pipeline

    def _unload_shape(self):
        if self.pipeline is not None:
            logger.info("Unloading Shape pipeline")
            pipe = self.pipeline
            self.pipeline = None
            del pipe
        release_cuda_memory()

    def _unload_paint(self):
        if self.paint_pipeline is not None:
            logger.info("Unloading Paint pipeline")
            pipe = self.paint_pipeline
            self.paint_pipeline = None
            del pipe
        release_cuda_memory()

    def get_queue_length(self):
        # asyncio.Semaphore is only a legacy constructor argument; GPU work is serialized
        # by _gpu_lock. Avoid depending on private semaphore internals for correctness.
        return 0

    def get_status(self):
        return {"speed": 1, "queue_length": self.get_queue_length()}

    @staticmethod
    def _notify(params, stage: str, progress: int):
        callback = params.get("_status_callback")
        if callable(callback):
            try:
                callback(stage, progress)
            except Exception:
                pass

    @torch.inference_mode()
    def generate(self, uid, params):
        with self._gpu_lock:
            return self._generate_sequential(uid, dict(params))

    def _generate_sequential(self, uid, params):
        started = time.time()
        uid_str = str(uid)
        logger.info("Generating 3D model for uid: %s", uid_str)
        self._notify(params, "preparing", 2)

        image = load_image_from_base64(params.get("image", ""))
        try:
            was_rgb = image.mode == "RGB"
            image = image.convert("RGBA")
            if was_rgb and params.get("remove_background", True):
                image = self.rembg(image.convert("RGB")).convert("RGBA")

            initial_path = os.path.join(self.save_dir, f"{uid_str}_initial.glb")

            # Stage 1: Shape
            self._notify(params, "shape", 5)
            shape_pipe = None
            try:
                shape_pipe = self._load_shape()
                generator = torch.Generator(device=self.device).manual_seed(
                    int(params.get("seed", 1234))
                )
                mesh = shape_pipe(
                    image=image,
                    generator=generator,
                    octree_resolution=params.get("octree_resolution", 256),
                    num_inference_steps=params.get("num_inference_steps", 5),
                    guidance_scale=params.get("guidance_scale", 5.0),
                    num_chunks=params.get("num_chunks", 8000),
                    output_type="trimesh",
                )[0]
                mesh.export(initial_path)
                del mesh
                del generator
                logger.info("Shape stage completed: %s", initial_path)
            except Exception as exc:
                logger.exception("Shape generation failed")
                raise ValueError(f"Failed to generate 3D mesh: {exc}") from exc
            finally:
                # Drop the local reference as well as self.pipeline before CUDA cleanup.
                shape_pipe = None
                self._unload_shape()

            # Original /generate supports Shape-only mode. The new combined endpoint
            # forces texture=True, while legacy callers can still request texture=False.
            if not params.get("texture", False):
                self._notify(params, "completed", 100)
                logger.info("Shape-only generation finished in %.2fs", time.time() - started)
                return initial_path, uid

            # Stage 2: Paint
            self._notify(params, "paint", 55)
            paint_pipe = None
            textured_obj = None
            try:
                paint_pipe = self._load_paint()
                output_obj = os.path.join(self.save_dir, f"{uid_str}_texturing.obj")
                textured_obj = paint_pipe(
                    mesh_path=initial_path,
                    image_path=image,
                    output_mesh_path=output_obj,
                    save_glb=False,
                )
                if not textured_obj or not os.path.isfile(textured_obj):
                    raise RuntimeError(f"Paint did not produce an OBJ file: {textured_obj}")

                temp_glb = os.path.join(self.save_dir, f"{uid_str}_texturing.glb")
                quick_convert_with_obj2gltf(textured_obj, temp_glb)
                final_path = os.path.join(self.save_dir, f"{uid_str}_textured.glb")
                os.replace(temp_glb, final_path)
            except Exception as exc:
                logger.exception("Texture generation failed")
                raise ValueError(f"Paint generation failed: {exc}") from exc
            finally:
                # Paint holds many CUDA module references; clear the local too.
                paint_pipe = None
                textured_obj = None
                self._unload_paint()

            self._notify(params, "completed", 100)
            logger.info("Shape + Paint generation finished in %.2fs", time.time() - started)
            return final_path, uid
        finally:
            try:
                image.close()
            except Exception:
                pass
            release_cuda_memory()

    def close(self):
        with self._gpu_lock:
            self._unload_shape()
            self._unload_paint()
