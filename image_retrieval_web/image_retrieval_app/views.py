# from django.http import HttpResponse
from django.template import RequestContext
# from django.template import loader
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.shortcuts import render_to_response
#from .form import ImageUploadForm
#from .models import Image
from django.conf import settings
import os

from .caffe_models.classifier import classifier



def index(request):
	if request.method == 'POST':
	    model = setup_classifer()
	    img_obj = request.FILES['image_file']
	    file_name = os.path.join(settings.MEDIA_ROOT, "images", img_obj.name)
	    if not os.path.isfile(file_name):
	        handle_uploaded_file(img_obj, file_name)
	    data = model.classify(file_name)
	    context = {'lables': data, 'image_file': '/media/images/'+img_obj.name}
	    return render(request, 'image_app.html', context)
	else:
	    return render(request, 'image_app.html')


def handle_uploaded_file(f, location):
    with open(location, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def setup_classifer():
    MODEL_ROOT = os.path.join(settings.BASE_DIR, "image_retrieval_app", 'caffe_models', 'model_files')
    deployment_model = os.path.join(MODEL_ROOT, "model_run.prototxt")
    trained_model = "image.jpg"
    caffe_model = os.path.join(MODEL_ROOT, trained_model)
    img_mean = os.path.join(MODEL_ROOT, "image_mean.npy")
    classifier_instance = classifier(deployment_model, caffe_model, img_mean)
    return classifier_instance


def image_app(request):
    if request.method == 'POST':
        model = setup_classifer()
        img_obj = request.FILES['image_file']
        file_name = os.path.join(settings.MEDIA_ROOT, "images", img_obj.name)
        if not os.path.isfile(file_name):
            handle_uploaded_file(img_obj, file_name)
        data = model.classify(file_name)
        context = {'class_label': data, 'image_file': '/media/images/'+img_obj.name}
        return render(request, 'caffeApp/image_app.html', context)
    else:
        return render(request, 'caffeApp/image_app.html')


def list(request):
    # Handle file upload
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = Image(imagefile=request.FILES['imagefile'])
            image.save()

            # Redirect to the document list after POST
            return HttpResponseRedirect(reverse('caffeApp:image_list'))
    else:
        form = ImageUploadForm()  # A empty, unbound form

    # Load documents for the list page
    images = Image.objects.all()

    # Render list page with the documents and the form
    return render_to_response(
        'caffeApp/list.html',
        {'documents': images, 'form': form},
        context_instance=RequestContext(request)
    )
