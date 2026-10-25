"""Pydantic schemas for structured model output."""

from typing import List

from pydantic import BaseModel, Field


class DishOutput(BaseModel):
    dish_name: str = Field(min_length=1)
    ingredients: List[str] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)
    cooking_time: str = ""
    difficulty_level: str = ""
    cost_estimation: str = ""
    suggested_price: str = ""
    nutritional_benefits: List[str] = Field(default_factory=list)
    main_image_url: str = ""
    core_selling_points: List[str] = Field(default_factory=list)
    marketing_title: str = ""
    marketing_description: str = ""
    tags: List[str] = Field(default_factory=list)
    image_suggestion: str = ""
    tavily_reference_images: List[str] = Field(default_factory=list)
    reference_image_error: str = ""
    generated_image_url: str = ""
    generated_image_error: str = ""


class RecipeOutput(BaseModel):
    dishes: List[DishOutput] = Field(min_length=1)
