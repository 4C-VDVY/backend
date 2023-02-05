from django.urls import path
from . import views

urlpatterns=[
    path('',views.overview,name="overview"),
    path('prod/',views.product_view),
    path('enter/',views.enter_details),
]