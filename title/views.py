from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import Product_detailsSerializer
from .models import Product_details
import json

# Create your views here.

@api_view(['GET'])
def overview(req):
    return Response("Running")

@api_view(['GET'])
def product_view(req):
    product=Product_details.objects.all()
    serializer=Product_detailsSerializer(product,many=True)
    return Response(serializer.data)

@api_view(['POST'])
def enter_details(req):
    serializer=Product_detailsSerializer(data=req.data)
    if(serializer.is_valid()):
        print("hi")
    dict={}
     # place your model here
    slogan="Lorem ipsum dolor sit amet, consectetur adipiscing"
    summary="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Cras erat dui, finibus vel lectus ac, pharetra dictum odio. Etiam risus sapien, auctor eu volutpat"
    dict["slogan"]=slogan
    dict["summary"]=summary
   
    json_data=json.dumps(dict)
    result=json.loads(json_data)
    return Response(result)

