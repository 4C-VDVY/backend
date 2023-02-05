from django.shortcuts import render
import folium
import geocoder
from .models import login
# Create your views here.
val=["India","Pakistan"]
def index(req):
    
    m=folium.Map(location=[20,78],zoom_start=4)
    for i in range(len(val)):
        # print(val[i])
        location=geocoder.osm(val[i])
        lat=location.lat
        long=location.lng
        country=location.country
        folium.Marker(location=[lat,long],tooltip='Click',popup=country).add_to(m)
    m=m._repr_html_()
    context={
        'm':m,
    }
    return render(req,'index.html',context)

def loginn(req):
    if(req.method=='POST'):
        uidd=req.POST['uname']
        passwordd=req.POST['pass']
        user=login(uid=uidd,password=passwordd)
        user.save()
    return render(req,'login.html')