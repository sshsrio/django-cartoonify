from django.shortcuts import render
from .forms import ImageForm
from .models import Image
import numpy as np
import cv2


def convert_to_grayscale(image):
    img = cv2.imread(image.path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(image.path, gray)


def convert_to_cartoon(image):
    img = cv2.imread(image.path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )

    color = cv2.bilateralFilter(img, 9, 300, 300)

    cartoon = cv2.bitwise_and(color, color, mask=edges)

    cv2.imwrite(image.path, cartoon)


# def home(request):
#     if request.method == "POST":
#         form = ImageForm(request.POST, request.FILES)
#         if form.is_valid():
#             form.save()
#     form = ImageForm()

#     img = Image.objects.all()
#     return render(request, "myapp/home.html", {"img": img, "form": form})


def home(request):
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            image_instance = form.save(commit=False)
            image_instance.save()
            convert_to_cartoon(image_instance.photo)
    else:
        form = ImageForm()

    img = Image.objects.all()
    return render(request, "myapp/home.html", {"img": img, "form": form})
