from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import Product_detailsSerializer
from .models import Product_details
from .generator import *
import json
import pandas as pd                        
from pytrends.request import TrendReq
from django.views.decorators.csrf import csrf_exempt
@csrf_exempt


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
    
    dict={}
    # serializer=Product_detailsSerializer(data=req.data)
    # if(serializer.is_valid()):
    #     print("hi")
    data=req.data
    title=data["title"]
    # summary=data["summary"]
    slogan=generate_slogan(title)
    dict["slogan"]=slogan
    
    # summary=generate_summary(summary)
    # dict["summary"]=summary
   
    json_data=json.dumps(dict)
    result=json.loads(json_data)
    return Response(result)

@api_view(['POST'])
def get_trends(req):
    title=req.data["title"]
    title=title.split()
    pytrend = TrendReq()
    pytrend.build_payload(kw_list=title)
    df = pytrend.interest_by_region()
    df = df[ (df[title[0]] >= 15) & (df[title[1]] >= 15) ]
    cd=df.head(10).to_dict()
    return Response(cd)

