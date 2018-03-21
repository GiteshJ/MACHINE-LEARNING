import UI_manager as u
from django.shortcuts import render
from django.db.models import Max
from django.http import JsonResponse
from django.http import HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage



def pred(request):
    if request.method == 'POST' and request.FILES['chooseFile']:
        myfile = request.FILES['chooseFile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
    return JsonResponse({"out" : u.predictcancer("media/" + filename)})

@csrf_exempt
def home(request):
    return render(request , "index.html")