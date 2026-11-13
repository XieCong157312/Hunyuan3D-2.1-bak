"""Pydantic request/response models for the Hunyuan3D-2.1 API."""
from typing import Optional
from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    """JSON request used by the original /generate and /send endpoints."""
    image: str = Field(..., description="Base64-encoded input image")
    remove_background: bool = Field(True, description="Remove background for RGB inputs")
    texture: bool = Field(False, description="Run the Paint stage after Shape")
    seed: int = Field(1234, ge=0, le=2**32 - 1)
    octree_resolution: int = Field(256, ge=64, le=512)
    num_inference_steps: int = Field(5, ge=1, le=20)
    guidance_scale: float = Field(5.0, ge=0.1, le=20.0)
    num_chunks: int = Field(8000, ge=1000, le=20000)
    face_count: int = Field(40000, ge=1000, le=100000)


class GenerationResponse(BaseModel):
    uid: str = Field(..., description="Unique generation task identifier")


class StatusResponse(BaseModel):
    status: str = Field(..., description="processing, texturing, completed or error")
    model_base64: Optional[str] = Field(None, description="Base64 GLB; retained for compatibility")
    message: Optional[str] = Field(None, description="Error message, when status is error")
    stage: Optional[str] = Field(None, description="Current pipeline stage")
    progress: Optional[int] = Field(None, description="Approximate stage progress")
    result_url: Optional[str] = Field(None, description="URL for downloading the finished GLB")


class HealthResponse(BaseModel):
    status: str
    worker_id: str
