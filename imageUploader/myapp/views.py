from django.shortcuts import render
from .forms import ImageForm
from .models import Image
import numpy as np
import cv2


def convert_to_cartoon(image):
    img = cv2.imread(image.path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(gray_img, 5)

    edges = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cartoon = cv2.bitwise_and(color_img, color_img, mask=edges)
    imgf = np.float32(cartoon).reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    compactness, label, center = cv2.kmeans(
        imgf, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    center = np.uint8(center)
    final_img = center[label.flatten()]
    final_img = final_img.reshape(cartoon.shape)
    cv2.imwrite(image.path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))


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
