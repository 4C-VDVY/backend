from rest_framework import serializers
from .models import Product_details

class Product_detailsSerializer(serializers.ModelSerializer):
    class Meta:
        model=Product_details
        fields='__all__'


