from django.urls import path
from . import views

urlpatterns=[
    path('',views.overview,name="overview"),
    path('prod/',views.product_view),
    path('slogan/',views.enter_details),
    path('trends/',views.get_trends),
]