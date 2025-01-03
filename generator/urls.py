from django.urls import path
from . import views

urlpatterns = [
    path("generate-test-outline", views.generate_test_outline, name="generate_test_outline"),
    path("generate_requirement", views.generate_requirement, name="generate_requirement")
]