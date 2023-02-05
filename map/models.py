from django.db import models

# Create your models here.
class login(models.Model):
    uid = models.CharField(max_length = 200)
    password = models.CharField(max_length = 200)
