from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DetectionBase(BaseModel):
    text: str = Field(..., max_length=32)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()


class DetectionCreate(DetectionBase):
    file_id: int
    source: str = "manual"


class DetectionUpdate(BaseModel):
    text: Optional[str] = Field(default=None, max_length=32)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    bbox_x1: Optional[float]
    bbox_y1: Optional[float]
    bbox_x2: Optional[float]
    bbox_y2: Optional[float]
    source: Optional[str]

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() if value else value


class DetectionRead(DetectionBase):
    id: int
    file_id: int
    source: str
    version: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class FileBase(BaseModel):
    original_filename: str
    status: str
    need_upscale: bool
    used_upscale: bool
    blur_score: Optional[float]
    contrast_score: Optional[float]
    width: Optional[int]
    height: Optional[int]
    notes: Optional[str]


class FileRead(FileBase):
    id: int
    processed_path: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class FileDetail(FileRead):
    detections: List[DetectionRead] = Field(default_factory=list)


class ExportRequest(BaseModel):
    file_ids: Optional[List[int]] = None
    include_manual_only: bool = False


class ExportStatus(BaseModel):
    archive_name: str
    total_files: int
    total_detections: int
    download_url: str
